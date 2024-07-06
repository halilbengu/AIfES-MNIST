''' 
Downloads train and test datasets and writes them into two header files as a 1-dimensional array.
'''

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Number of training images", type=int, default=60000)
parser.add_argument("--test", help="Number of test images", type=int, default=10000)
parser.add_argument("--batch", help="Batch Size while training", type=int, default=1000)
parser.add_argument("--epoch", help="Epochs while training", type=int, default=10)
args = parser.parse_args()

# Load and preprocess the MNIST data set
print(f"\nDownloading MNIST Dataset: #Training:{args.train} , #Test:{args.test} \n")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(float)/255.0
x_test = x_test.astype(float)/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

NUM_TRAINING_DATA = args.train   # Max 60000
NUM_TEST_DATA = args.test       # Max 10000

def generate_train_data():

    with open("MNIST_training_data.h", "w") as f:    
        f.write("float MNIST_training_data[] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write(",\n")
            x_train_flatten = x_train[i].flatten()
            f.write(str(x_train_flatten[0]) + "f")
            for j in range(1,784):
                f.write(", " + str(x_train_flatten[j]) + "f")
        f.write("\n};")

    with open("MNIST_training_data_label.h", "w") as f:    
        f.write("float MNIST_training_data_label[] = {\n")
        for i in range(0,NUM_TRAINING_DATA):
            if i != 0:
                f.write(",\n")
            f.write( str(y_train[i][0]) + "f")
            for j in range(1,10):
                f.write(", " + str(y_train[i][j]) + "f")
        f.write("\n};")


def generate_test_data():

    with open("MNIST_test_data.h", "w") as f:    
        f.write("float MNIST_test_data[] = {\n")
        for i in range(0,NUM_TEST_DATA):
            if i != 0:
                f.write(",\n")
            x_test_flatten = x_test[i].flatten()
            f.write(str(x_test_flatten[0]) + "f")
            for j in range(1,784):
                f.write(", " + str(x_test_flatten[j]) + "f")
        f.write("\n};")


    with open("MNIST_test_data_label.h", "w") as f:    
        f.write("float MNIST_test_data_label[] = {\n")
        for i in range(0,NUM_TEST_DATA):
            if i != 0:
                f.write(",\n")
            f.write(str(y_test[i][0]) + "f")
            for j in range(1,10):
                f.write(", " + str(y_test[i][j]) + "f")
        f.write("\n};")
        f.write(f"\n#define BATCH_SIZE {args.batch}")
        f.write(f"\n#define EPOCHS {args.epoch}")


generate_train_data()
generate_test_data()