"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from collections import deque

INPUT_STREAM = "resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
PEDEST_MODEL = "model/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.xml"
SSD_INCEPTION = "model/ssd_inception_v2_coco_2018_01_28/FP16/ssd_inception_v2_coco_2018_01_28.xml"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str, default=SSD_INCEPTION,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str, default=INPUT_STREAM,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=CPU_EXTENSION,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


# The callback for when the client receives a response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("person")
    client.subscribe("person/duration")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, port=MQTT_PORT, keepalive=MQTT_KEEPALIVE_INTERVAL)
    client.on_connect

    return client


#Draw a round target on the located personne
def draw_target(frame,xmin,ymin,xmax,ymax):
    xc = int((xmax + xmin) /2)
    yc = int((ymax + ymin) /2)
    
    cv2.circle(frame,(xc,yc),20,(0,0,255),cv2.FILLED)
    
    return None

## Draw boxes around people
def draw_boxes(frame, result, width, height, threshold, infertime):
    '''
    Draw bounding boxes onto the frame.
    ''' 
    count_p = 0
    label_conf = 0

    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        label = box[1]
        
        if conf >= threshold:
            #label_conf = label
            #print("label conf draw box func: ", label) 
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            draw_target(frame,xmin,ymin,xmax,ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            cv2.putText(frame, str(label), (xmax,ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, str( " Inference time: "+ '{0:.2f}'.format(infertime) + " ms"), (70,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 1)
            #label_conf = count_people_label(label)
            count_p +=1
        
        #print("label draw box func: ", label) 

    return frame, count_p



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    #Variables initial state 
 
    CURRENT_COUNT_PEOPLE = 0
    PREVIOUS_COUNT = 0
    TOTAL_PEOPLE = 0
    STAY_TIME = 0
    maxlen = 24
    
    # queue to accumulate last "max_len" number of detections
    tracking_p = deque(maxlen=maxlen)


    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    PROB_THRESHOLD = args.prob_threshold
    MODEL = args.model
    DATA_INPUT = args.input
    CPU_EXTENSION = args.cpu_extension
    DEVICE = args.device

    ###Load the model through `infer_network` ###
    infer_network.load_model(MODEL,DEVICE,CPU_EXTENSION)
    inputshape = infer_network.get_input_shape()

    ### Handle the input stream ###

    if DATA_INPUT =='CAM':
        input_stream = 0
        single_image = False
    elif DATA_INPUT[-4:] in [".jpg", ".png"]:
        single_image = True
        input_stream = DATA_INPUT
    else:
        single_image=False
        input_stream = DATA_INPUT
        assert os.path.isfile(input_stream)
    
    # Get and open video capture
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    #print("inputshape: ", inputshape, "video input -> width:", width, " height:", height)

    ### Loop until stream is over ###

    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        

        ### Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (inputshape[3], inputshape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### set start infer time
        infer_start=time.time()

        ###  Start asynchronous inference for specified request ###
        infer_network.async_inference(p_frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            ###Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### infer timing
            log.info(f"infer time for one frame = {time.time()-infer_start} seconds")

            infertime = time.time()-infer_start
            ### TODO: Extract any desired stats from the results ###
            
            infertime = (time.time()-infer_start) * 1000 
            out_frame , label_conf =  draw_boxes(frame, result, width, height, PROB_THRESHOLD,infertime)


            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            # append number of detections to "tracking_p" queue
            tracking_p.append(label_conf)
            # proportion of frames with a positive detection 
            num_tracked = 0
            if np.sum(tracking_p)/maxlen > 0.1:
                num_tracked = 1
            
            if num_tracked > PREVIOUS_COUNT:
                start_time = time.time()
                num_persons_in = num_tracked - PREVIOUS_COUNT
                TOTAL_PEOPLE += num_persons_in
                PREVIOUS_COUNT = num_tracked
                client.publish("person", json.dumps({"count":PREVIOUS_COUNT, "total":TOTAL_PEOPLE}), retain=True)
            
            if num_tracked < PREVIOUS_COUNT:
                PREVIOUS_COUNT = num_tracked

            if num_tracked > 0:
                STAY_TIME += (time.time() - start_time)

            if TOTAL_PEOPLE > 0:
                mean_stay_time = STAY_TIME/TOTAL_PEOPLE
                client.publish("person/duration", json.dumps({"duration": int(mean_stay_time)}), retain=True)

        ### Write an output image if `single_image_mode` ###
        ### Send the frame to the FFMPEG server ###
        if single_image:
            cv2.imwrite("output.jpg", out_frame)
        else :
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
