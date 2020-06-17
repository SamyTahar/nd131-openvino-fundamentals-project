## Explaining Custom Layers in OpenVINO™
What is the process of custom layer? 
What are the reasons to use custom layers?

When the model is converted by openvino it will check if the layer exists in the predefined layers. If not we need to create on according to the layer operation that we need to perform. The predefine layers depends also on the hardware used for example we could use a layer that is compatible with CPU but not with VPU it is possible to create a custom layer or use a method to use the CPU when the incompatible layer is called.

First adding a custom layer depends on the framework used to create the model (Caffe, TensorFlow, MXNET)

There are 2 options the first one for both of these frameworks custom layers can be loaded as extensions in the model optimizer the second options will be for Caffe to register the layer as Custom and use Caffe to calculate the layer then for TensorFlow when subgraph that is not compatible we will change it for a compatible one without the need to have TensorFlow installed. 

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

## Compare Model Performance
I have converted from tensorflow the ssd_inception_v2_coco model with a speed of 42 ms and COCO mAP (mean average precision) of 24.

Here is the link to the model 
http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz

Here is the command that I have used to convert the model:
$MOD_OPT/mo.py --input_model frozen_inference_graph.pb --data_type FP16 --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

The model doesn't seem to work properly the speed is on the inferred model between 50 and 100 ms.  The threshold has been set to 0.1 to get the most of this model however one person is difficult to be found by the model I am guessing that is it because this person is not moving a lot compare to the other persons. As a result, the number of persons on the screen increase compares to the ground truth.

I have used another model, the Pedestrian model from openvino model zoo with a threshold of 0.9 I can get a speed between 25 and 40 ms and the persons are well recognized. 

python3 benchmark_app.py -m <ir_dir>/googlenet-v1.xml -d CPU -api async -i <INSTALL_DIR>/deployment_tools/demo/car.png --progress true -b 1

I could have used the following tool benchmark_app.py to get a benchmark of the model however some error is appearing "ModuleNotFoundError: No module named 'openvino.tools' " whereas I can perform other tasks.

python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/benchmark_tool/benchmark_app.py -m -CPU api async -i resources/img_test.png

Note: OpenVino tool kit install could be better.

On one hand in the cloud, model computation is not an issue however could cost a lot on second hand the model computation on edge is critical as we want to operate our model au low latency hardware and maximize the model performance.  

A great capability is to be able to use all compute power available to distribute the computation model on CPU, GPU, VPU at the same time very handy when you don't want to deal with custom layers on VPU and that are compatible on CPU or GPU.


## Assess Model Use Cases
The people counter app could be useful in marketing to know how many people are in a mall, for example, or when people are inside a shop we can evaluate the sales potential or optimize the product shelf. 

This App can be useful in a security camera to evaluate if the area has a lot of crowds and compared it with previous records if the area is not supposed to be packed with a lot of crowds then maybe something is happening. 

Assess Effects on End-User Needs
The model accuracy is mainly due to the data that has been feed to it.
If the dataset is composed of high exposure images then the model is feed by a low exposure image the accuracy will be bad.

The dataset should be well distributed with low exposure, high exposure, rotation on the left-right, and so on.

Another this that has to be taken into account is the camera lens distortion. Each camera as a lens distortion that has to be compensated OpenCV can help undistorted the image from the camera by finding the amount of distortion.

Image size is also important when the model is trained, the higher is the resolution the harder/longer will be to compute the model. When using the model the camera image has to be feed with the image size that the model has been trained on.

If the model is converted with a high-resolution image as a feed the computation will be intensive and not suitable on the edge whereas we want to be able to work on low latency and low computation as possible.

 ## Model Research NONE

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...