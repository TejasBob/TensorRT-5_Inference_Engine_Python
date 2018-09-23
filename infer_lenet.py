import tensorflow as tf
import numpy as np 
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit



MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")

img, label = MNIST_DATASETS.test.next_batch(1)
img = img[0]
img = img.astype(np.float32)
label = label[0]

model_file = "model_data/mnist.uff"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


builder = trt.Builder(TRT_LOGGER)  
network = builder.create_network() 

with trt.UffParser() as parser:


    parser.register_input("Placeholder", (1, 28, 28))
    parser.register_output("fc2/Softmax")
    parser.parse(model_file, network)


    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 10


    # h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)

    # h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)

    # d_input = cuda.mem_alloc(h_input.nbytes)

    # d_output = cuda.mem_alloc(h_output.nbytes)


    with builder.build_cuda_engine(network) as engine:
        output = np.empty(10, dtype = np.float32)


        # Alocate device memory
        d_input = cuda.mem_alloc(1 * img.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)
        bindings=[int(d_input), int(d_output)]

        stream = cuda.Stream()

        with engine.create_execution_context() as context:
            cuda.memcpy_htod_async(d_input, img, stream)

            context.execute_async(bindings = bindings, stream_handle=stream.handle)

            cuda.memcpy_dtoh_async(output, d_output, stream)

            stream.synchronize()

            print("true label : ", label)
            print(np.argmax(output))
            print(output)



