import numpy as np
import time
import re
import random
from random import shuffle

# vector files
cn_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\ChineseVectors.txt'
jp_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\JapaneseVectors.txt'
eng_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\EnglishVectors.txt'
# endpoint files
cn_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\ChineseEndpoints.txt'
jp_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\JapaneseEndpoints.txt'
eng_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\EnglishEndpoints.txt'
# counter, for keeping track of how many items
run_counter_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\RunCount.txt'
# the output, randomized data used for training
output_file_vector = 'C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataRandomized\\RTrainingVectors1.txt'
output_file_endpoints = 'C:\\Users\\linka\\OneDrive\\Desktop\Code\HuiFu\\TrainingDataRandomized\\RTrainingEndpoints1.txt'

# first item is the input file, second is result file, third is a vector representing how many items in each run for that language
vector_file_mappings = {
    1: (jp_vectors_file, jp_vectors_endpoints_file, []),
    2: (cn_vectors_file, cn_vectors_endpoints_file,[]),
    0: (eng_vectors_file, eng_vectors_endpoints_file,[])
}

# load the counts into the vector file mappings
def load_counters() :
    with open(run_counter_file, "r") as r:
        for line in r:
            parts = line.strip().split(",")
            if len(parts) < 2:
                # Handle the case where the line doesn't have at least two values
                print(f"Skipping line: {line}")
                continue

            language_code = int(parts[0])
            counts = [int(x) for x in parts[1:]]  # Convert the rest of the values to integers

            if language_code in vector_file_mappings:
                vector_file_mappings[language_code][2].extend(counts)

load_counters()
for language_code, (vector_file, endpoints_file, vector_list) in vector_file_mappings.items():
    print(f"Language Code: {language_code}")
    print(f"Vector File: {vector_file}")
    print(f"Endpoints File: {endpoints_file}")
    print(f"Vector List: {vector_list}")
    print()

###### prio which run
###### what is the size of the output
###### what percentage is biased towards the priority
priority_run = 0
training_size = 20000
bias = 80 # bias towards new data
# a vector that represents how many data we use for each training done
def set_up_datapoints():
    total_runs = len(vector_file_mappings[0][2])
    priority_run_count = int(training_size * bias /100) # we prio the most recent run
    even_distribution_count = 0
    if not(total_runs == 1):
        even_distribution_count = (training_size - priority_run_count) // (total_runs - 1)
    data_points_vector = []
    for i in range(total_runs):
        if i == priority_run:
            data_points_vector.append(priority_run_count)
        else:
            data_points_vector.append(even_distribution_count)
    print("Data Points Vector:", data_points_vector)
    return data_points_vector

def combine_runs(): 
    # Create a list to store vectors organized by 
    # each item in the list is an entire run
    num_runs = len(vector_file_mappings[0][2])
    vectors_by_run = []
    print("num_runs: " + str(num_runs))
    # Read the vector and endpoint files and populate data_by_run
    for run in range(num_runs):
        combined_run_data = {"vectors": [], "endpoints": []}
        for language_code, (vector_file, endpoint_file, language_counts) in vector_file_mappings.items():
            count = language_counts[run]
            with open(vector_file, "r") as vector_file, open(endpoint_file, "r") as endpoint_file:
                vectors = [line.strip().split() for line in vector_file][:count]
                # Parse the items in the "endpoints" file as lists of integers
                endpoints = [list(map(int, line.strip()[1:-1].split(','))) for line in endpoint_file][:count]
                combined_run_data["vectors"].extend(vectors)
                combined_run_data["endpoints"].extend(endpoints)
        vectors_by_run.append(combined_run_data)
        
        # Print the length of vectors and endpoints for the current run
        print(f"Run {run + 1} - Vectors: {len(combined_run_data['vectors'])}, Endpoints: {len(combined_run_data['endpoints'])}")
        
    return vectors_by_run


# given x input and y amount, randomly select y amount of x and put into a vector
def random_sample(input, amount):
    if(len(input) <= amount):
        return input
    return random.sample(input, amount)

def randomize_all_runs(data_points, all_runs):
    # Initialize a list to store the randomly selected items
    selected_items = []

    # Iterate through data_points and all_runs in parallel
    for data_point, run in zip(data_points, all_runs):
        # Randomly sample 'data_point' items from the current run
        selected = random_sample(run, data_point)
        selected_items.extend(selected)

    return selected_items

def randomize_runs_go(data_points):
    # each entry in this vector has the structure of {"vectors": [vector1, vector2, ...], "endpoints": [endpoint1, endpoint2, ...]},
    vectors_by_run = combine_runs()
    randomized_data = randomize_all_runs_go(data_points, vectors_by_run)
    return randomized_data

def randomize_all_runs_go(data_points, vectors_by_run):
    # Initialize a list to store the randomly selected items
    selected_items = []

    # Iterate through data_points and vectors_by_run in parallel since vectors by run matches datapoints
    for data_point, run in zip(data_points, vectors_by_run):
        # Combine vectors and endpoints for the current run as tuples
        combined_data = list(zip(run["vectors"], run["endpoints"]))
        
        # Randomly shuffle the combined data
        shuffle(combined_data)
        
        # Take the desired number of items from the shuffled data
        selected = combined_data[:data_point]
        
        selected_items.extend(selected)

    return selected_items

# write the tuples into the two different files
def write_vectors_and_endpoints(selected_items):
    # Open the output files for vectors and endpoints
    with open(output_file_vector, "w") as vector_file, open(output_file_endpoints, "w") as endpoint_file:
        for vector, endpoint in selected_items:
            # Convert vector and endpoint to space-separated strings
            vector_str = " ".join(str(value) for value in vector)
            endpoint_str = " ".join(str(value) for value in endpoint)

            # Write the vector and endpoint to their respective files
            vector_file.write(vector_str + "\n")
            endpoint_file.write(endpoint_str + "\n")

dp_vector = set_up_datapoints()
randomized_data = randomize_runs_go(dp_vector)
write_vectors_and_endpoints(randomized_data)

