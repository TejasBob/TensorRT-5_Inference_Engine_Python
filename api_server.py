import numpy as np 
import tensorrt as trt 
import pycuda.driver as cuda

from flask import Flask, url_for
from flask import request
from flask import json
import cv2
import time
import tensorflow as tf


app = Flask(__name__)

MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")

BATCH_SIZE = 4096
print("batch size : ", BATCH_SIZE)


img, label = MNIST_DATASETS.test.next_batch(BATCH_SIZE)
img = img[:]
img = img.reshape((1, BATCH_SIZE * 784))
img = img.astype(np.float32)
label = label[:]

print("labels : ", label)
print("img shape : " , img.shape)
print('size : ', img.nbytes / 1024 /1024)

model_file = "model_data/mnist.uff"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)  
network = builder.create_network() 

parser =  trt.UffParser()
parser.register_input("Placeholder", (1, 28, 28))
parser.register_output("fc2/Relu")
parser.parse(model_file, network)
builder.max_batch_size = BATCH_SIZE
builder.max_workspace_size = 1 << 20


@app.route("/predict", methods = ["POST"])
def predict():

    time1 = time.time()
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    time2 = time.time()
    print("time to get context : ", time2 - time1)

    with builder.build_cuda_engine(network) as engine:
        output = np.empty(10 * BATCH_SIZE, dtype = np.float32)

        d_input = cuda.mem_alloc(1 * img.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)
        bindings=[int(d_input), int(d_output)]

        stream = cuda.Stream()

        with engine.create_execution_context() as context:
            cuda.memcpy_htod_async(d_input, img, stream)

            context.execute_async(bindings = bindings, stream_handle=stream.handle, batch_size = BATCH_SIZE)

            cuda.memcpy_dtoh_async(output, d_output, stream)

            stream.synchronize()

            # print("true label : ", label)

            result = []
            accuracy = np.zeros((1, BATCH_SIZE), np.uint8)
            for ii in range(BATCH_SIZE):
                result.append(np.argmax(output[ii*10:(ii+1)*10]))
                if result[ii] == label[ii]:
                    accuracy[0, ii] = 1
                # print(output[ii*10:(ii+1)*10])
            # print(result)
            print("accuracy : ", np.sum(accuracy) / BATCH_SIZE)

    ctx.pop()
    return "Done\n"#str(output)



if __name__ == '__main__':
    app.run()