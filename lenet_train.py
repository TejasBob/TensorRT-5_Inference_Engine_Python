import tensorflow as tf
import numpy as np 
import time 
import os
import uff
import tensorrt as trt 

import pycuda.driver as cuda
import pycuda.autoinit



STARTER_LEARNING_RATE = 1e-4
BATCH_SIZE = 10
NUM_CLASSES = 10
MAX_STEPS = 3000
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2
OUTPUT_NAMES = ["fc2/Relu"]

def WeightsVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, name='weights'))

def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='biases'))

def Conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    filter_size = W.get_shape().as_list()
    pad_size = filter_size[0]//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    x = tf.pad(x, pad_mat)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def MaxPool2x2(x, k=2):
    # MaxPool2D wrapper
    pad_size = k//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')



def network(images):
    # Convolution 1
    with tf.name_scope('conv1'):
        weights = WeightsVariable([5,5,1,32])
        biases = BiasVariable([32])
        conv1 = tf.nn.relu(Conv2d(images, weights, biases))
        pool1 = MaxPool2x2(conv1)

    # Convolution 2
    with tf.name_scope('conv2'):
        weights = WeightsVariable([5,5,32,64])
        biases = BiasVariable([64])
        conv2 = tf.nn.relu(Conv2d(pool1, weights, biases))
        pool2 = MaxPool2x2(conv2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Fully Connected 1
    with tf.name_scope('fc1'):
        weights = WeightsVariable([7 * 7 * 64, 1024])
        biases = BiasVariable([1024])
        fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Fully Connected 2
    with tf.name_scope('fc2'):
        weights = WeightsVariable([1024, 10])
        biases = BiasVariable([10])
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases)

    return fc2


def loss_metrics(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='softmax')
    return tf.reduce_mean(cross_entropy, name='softmax_mean')


def training(loss):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE,
                                               global_step,
                                               100000,
                                               0.75,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            summary):

    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        log, correctness = sess.run([summary, eval_correct], feed_dict=feed_dict)
        true_count += correctness
    precision = float(true_count) / num_examples
    tf.summary.scalar('precision', tf.constant(precision))
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    return log


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    return images_placeholder, labels_placeholder



def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
    feed_dict = {
        images_pl: np.reshape(images_feed, (-1,28,28,1)),
        labels_pl: labels_feed,
    }
    return feed_dict



def run_training(data_sets):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        logits = network(images_placeholder)
        loss = loss_metrics(logits, labels_placeholder)
        train_op = training(loss)
        eval_correct = evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log",
                                               graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter("/tmp/tensorflow/mnist/log/validation",
                                            graph=tf.get_default_graph())
        sess.run(init)
        for step in range(MAX_STEPS):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join("/tmp/tensorflow/mnist/log", "model.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                print('Validation Data Eval:')
                log = do_eval(sess,
                              eval_correct,
                              images_placeholder,
                              labels_placeholder,
                              data_sets.validation,
                              summary)
                test_writer.add_summary(log, step)
        # Return sess

        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graphdef,
                                                                    OUTPUT_NAMES)
        return tf.graph_util.remove_training_nodes(frozen_graph)


MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")
tf_model = run_training(MNIST_DATASETS)


uff_model = uff.from_tensorflow(tf_model, ["fc2/Relu"]) # this is a byte stream of model

f = open("mnist.uff", "wb")
f.write(uff_model)
f.close()

model_file = "/home/bob/Documents/flask/mnist.uff"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


builder = trt.Builder(TRT_LOGGER)  
network = builder.create_network() 
parser = trt.UffParser() 


parser.register_input("Placeholder", (1, 28, 28))
parser.register_output("fc2/Relu")
parser.parse(model_file, network)


builder.max_batch_size = 1
builder.max_workspace_size = 1 << 10

engine = builder.build_cuda_engine(network)

# serialized_engine = engine.serialize()

# with open("sample.engine", "wb") as f:
# 	f.write(engine.serialize())



img, label = MNIST_DATASETS.test.next_batch(1)
img = img[0]
img = img.astype(np.float32)
label = label[0]



# h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)

# h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)

# d_input = cuda.mem_alloc(h_input.nbytes)

# d_output = cuda.mem_alloc(h_output.nbytes)


output = np.empty(10, dtype = np.float32)


# Alocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)
bindings=[int(d_input), int(d_output)]

stream = cuda.Stream()

context = engine.create_execution_context() 

cuda.memcpy_htod_async(d_input, img, stream)

context.execute_async(bindings = bindings, stream_handle=stream.handle)

cuda.memcpy_dtoh_async(output, d_output, stream)

stream.synchronize()

print("true label : ", label)
print(output)

