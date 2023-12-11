import numpy as np
import re

# vector files
cn_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\ChineseVectors.txt'
jp_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\JapaneseVectors.txt'
eng_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\EnglishVectors.txt'
# endpoint files
cn_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\ChineseEndpoints.txt'
jp_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\JapaneseEndpoints.txt'
eng_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\EnglishEndpoints.txt'
# final files
final_vectors_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\AllVectors.txt'
final_vectors_endpoints_file = 'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized(t)\\AllEndpoints.txt'

# Function to load data from a vectors file and its corresponding endpoints file
def load_language_data(vectors_file, endpoints_file):
    with open(vectors_file, 'r') as f:
        vectors_lines = f.readlines()
    with open(endpoints_file, 'r') as f:
        endpoints_lines = f.readlines()

    # Ensure the number of vectors matches the number of endpoints
    if len(vectors_lines) != len(endpoints_lines):
        raise ValueError("The number of vectors and endpoints must match.")

    training_vectors = np.array([list(map(float, line.strip().split())) for line in vectors_lines])

    # Process endpoints: Remove brackets, split by comma, and convert to integers
    endpoint_pattern = re.compile(r'\[(.*?)\]')
    training_endpoints = np.array([
        list(map(int, endpoint_pattern.search(line).group(1).split(',')))
        for line in endpoints_lines
    ])

    return training_vectors, training_endpoints

# Define file paths for each language
languages = {
    'Chinese': (cn_vectors_file, cn_vectors_endpoints_file),
    'Japanese': (jp_vectors_file, jp_vectors_endpoints_file),
    'English': (eng_vectors_file, eng_vectors_endpoints_file)
}

# Initialize lists to store data for each language
all_training_vectors = []
all_training_endpoints = []

def clear_file(file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.truncate(0)  # Truncate the file to remove all content
        print(f"Cleared the content of {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load data for each language and concatenate
for language, (vectors_file, endpoints_file) in languages.items():
    vectors, endpoints = load_language_data(vectors_file, endpoints_file)
    all_training_vectors.append(vectors)
    all_training_endpoints.append(endpoints)

# Concatenate data for all languages
final_training_vectors = np.vstack(all_training_vectors)
final_training_endpoints = np.vstack(all_training_endpoints)

# clear both final vectors and endpoints
clear_file(final_vectors_file)
clear_file(final_vectors_endpoints_file)

# Save the final combined data to files

with open(final_vectors_file, 'w') as f:
    for vector in final_training_vectors:
        f.write(" ".join(map(str, vector)) + "\n")
with open(final_vectors_endpoints_file, 'w') as f:
    for endpoint in final_training_endpoints:
        f.write("[" + ", ".join(map(str, endpoint)) + "]\n")

