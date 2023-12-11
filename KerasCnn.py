import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import random
import re

input_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\AllVectors.txt'
input_vectors_endpoint = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\AllEndpoints.txt"
input_testing_vectors = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized300\\vectors.txt"
input_testing_endpoints = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized300\\endpoints.txt"

# load in training data
def load_training_set(vectors_file, endpoints_file):
    # Read training vectors from the vectors file
    with open(vectors_file, 'r') as f:
        vectors_lines = f.readlines()

    # Read training endpoints (one-hot encoded values) from the endpoints file
    with open(endpoints_file, 'r') as f:
        endpoints_lines = f.readlines()

    # Ensure the number of vectors matches the number of endpoints
    if len(vectors_lines) != len(endpoints_lines):
        raise ValueError("The number of vectors and endpoints must match.")

    # Parse vectors and endpoints into NumPy arrays
    training_vectors = np.array([list(map(float, line.strip().split())) for line in vectors_lines])
    # Parse and clean endpoints
    endpoint_pattern = re.compile(r'\[(.*?)\]')
    training_endpoints = np.array([
        list(map(int, re.sub(r'[,\[\]]', '', endpoint_pattern.search(line).group(1)).split()))
        for line in endpoints_lines
    ])

    return training_vectors, training_endpoints

# load in testing data
# randomly select 10k data from all vectors and endpoints
def load_testing_set(vector_file, endpoints_file, num_samples=10000):
    # Initialize lists to store data
    all_testing_vectors = []
    all_testing_endpoints = []

    # Load data from the vector file
    with open(vector_file, 'r') as f:
        vectors_lines = f.readlines()

    # Load data from the endpoints file
    with open(endpoints_file, 'r') as f:
        endpoints_lines = f.readlines()

    # Ensure the number of vectors matches the number of endpoints
    if len(vectors_lines) != len(endpoints_lines):
        raise ValueError("The number of vectors and endpoints must match.")

    # Parse vectors and endpoints into NumPy arrays
    testing_vectors = np.array([list(map(float, line.strip().split())) for line in vectors_lines])

    # Parse and clean endpoints
    endpoint_pattern = re.compile(r'\[(.*?)\]')
    testing_endpoints = np.array([
        list(map(int, re.sub(r'[,\[\]]', '', endpoint_pattern.search(line).group(1)).split()))
        for line in endpoints_lines
    ])

    # Shuffle the data
    combined_data = list(zip(testing_vectors, testing_endpoints))
    random.shuffle(combined_data)
    testing_vectors, testing_endpoints = zip(*combined_data)

    # Append a subset of the data to the final testing set
    num_samples_remaining = min(num_samples, len(testing_vectors))
    all_testing_vectors.extend(testing_vectors[:num_samples_remaining])
    all_testing_endpoints.extend(testing_endpoints[:num_samples_remaining])

    # Convert the final testing set to NumPy arrays
    final_testing_vectors = np.array(all_testing_vectors)
    final_testing_endpoints = np.array(all_testing_endpoints)

    return final_testing_vectors, final_testing_endpoints

# loads both training and testing, making 10k of randomized total data as the testing, the rest become training
def load_testing_set2(vector_file, endpoints_file, num_samples=10000):
    # Initialize lists to store data
    all_testing_vectors = []
    all_testing_endpoints = []

    # Load data from the vector file
    with open(vector_file, 'r') as f:
        vectors_lines = f.readlines()

    # Load data from the endpoints file
    with open(endpoints_file, 'r') as f:
        endpoints_lines = f.readlines()

    # Ensure the number of vectors matches the number of endpoints
    if len(vectors_lines) != len(endpoints_lines):
        raise ValueError("The number of vectors and endpoints must match.")

    # Parse vectors and endpoints into NumPy arrays
    testing_vectors = np.array([list(map(float, line.strip().split())) for line in vectors_lines])

    # Parse and clean endpoints
    endpoint_pattern = re.compile(r'\[(.*?)\]')
    testing_endpoints = np.array([
        list(map(int, re.sub(r'[,\[\]]', '', endpoint_pattern.search(line).group(1)).split()))
        for line in endpoints_lines
    ])

    # Shuffle the data
    combined_data = list(zip(testing_vectors, testing_endpoints))
    random.shuffle(combined_data)
    testing_vectors, testing_endpoints = zip(*combined_data)

    # Append a subset of the data to the final testing set
    # taking 0 - num_samples
    num_samples_remaining = min(num_samples, len(testing_vectors))
    all_testing_vectors.extend(testing_vectors[:num_samples_remaining])
    all_testing_endpoints.extend(testing_endpoints[:num_samples_remaining])
    #The rest become our training data
    remaining_testing_vectors = testing_vectors[num_samples_remaining:]
    remaining_testing_endpoints = testing_endpoints[num_samples_remaining:]

    # Convert the final testing set to NumPy arrays
    final_testing_vectors = np.array(all_testing_vectors)
    final_testing_endpoints = np.array(all_testing_endpoints)
    # Convert the final training set to NummPy arrays
    final_training_vectors = np.array(remaining_testing_vectors)
    final_training_endpoints = np.array(remaining_testing_endpoints)

    return final_training_vectors, final_training_endpoints, final_testing_vectors, final_testing_endpoints


def define_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(300,)))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu'))
    # add dropout?
    model.add(Dropout(0.6))
    model.add(Dense(6, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_model_quad_layer():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(300,)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    # add dropout?
    model.add(Dropout(0.4))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_model_l2normalization():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(300,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    opt = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history, plot_name):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(plot_name + '.png')
	pyplot.close()

model_dir = "doubleDropOut4Layers3dropout300.h5"
plot_name = "doubleDropOut4Layers3dropout300.h5"
def run_test_harness(mode):
    # load data set
    #trainX, trainY = load_training_set(input_vectors_file, input_vectors_endpoint)
    #testX, testY = load_testing_set(input_testing_vectors,input_testing_endpoints)
    trainX,trainY,testX,testY = load_testing_set2(input_testing_vectors, input_testing_endpoints)
    if(mode == 'load'):
        model = load_model(model_dir)
    elif(mode == 'new'):
        model = define_model_quad_layer()
    # fit model 
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history, plot_name)
    model.save(model_dir)
run_test_harness('new')
