# TensorRT-5_Inference_Engine_Python
TensorRT-5  based inference engine in Python

## Purpose
This is a POC project.
Aim of this project is to get acquinted with use of TensorRT api in Python3.

The project includes 
  
    1. Tensorflow script to train a Lenet Classifier.
    
    2. Python Webserver script to host a REST api to perform inference.
    
    3. Client script to do concurrent requests to REST api to check inference performance.
    

## Steps:

     $ python3 train_lenet.py
    

This will train a LeNet classifier, convert it to UFF format and save to disk

     $ python3 infer_lenet.py
   
   
This will help you to understand how to run the model using TensorRT.

      $ python3 api_server.py
   
This will host a flask server which will accept http POST reuests to perform inference. 

      $ python3 client.py
      
This script will do concurrent requests to REST api hosted on **localhost:5000/predict**.


Num of concurrent requests can be changed by changing **max_workers** in the script.


While working with Pycuda, I found that it takes around 2 sec send a POST request --> perform inference on 1 image -> return results.


Hence, to get higher throughput I changed the batch size from 1 to 1024, 2048, 4096 etc. 


**BATCH_SIZE** variable in api_server.py will help you set up thebatch size for inference.

Hence, for every POST request I send to the api, TensorRT is given BATCH_SIZE no of images to infer, making sure we get higher throughput.

On my laptop with **Nvidia 940MX** I was able to infer **4096 images (each of size 28*28)** in **7.762 sec**

     
## Installation:

TensorRT 5 can be downloaded from [here](https://developer.nvidia.com/tensorrt)

Installation instructions are present  : [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)

I highly recommend reading Developer Guide for TensorRT 5 before going through this project's code.





