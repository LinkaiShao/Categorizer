import numpy as np
from LAC import LAC
import time
import re

# Initialize Baidu LAC
lac = LAC(mode='rank')

def z_score_normalization(embedding):
    mean = np.mean(embedding)
    std_dev = np.std(embedding)
    standardized_embedding = (embedding - mean) / std_dev
    return standardized_embedding

def remove_non_alphabet_characters(input_string):
    # Use a regular expression to remove all non-alphabet characters
    cleaned_string = re.sub(r'[^a-zA-Z ]', '', input_string)
    return cleaned_string

def load_tencent_word_vectors(file_path):
    # Initialize an empty dictionary to store word vectors
    word_vectors = {}

    # Open and read the Tencent word embedding text file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line into word and vector parts
            parts = line.strip().split(' ')
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]], dtype='float32')

            # Store the word vector in the dictionary
            word_vectors[word] = vector

    return word_vectors

# Load Tencent word vectors
start_time = time.time()
tencent_word_vectors = load_tencent_word_vectors('C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\WordEmbedding\\tencentAi200\\tencentAi200\\tencentAiEmbedding200.txt')
end_time = time.time()
execution_time = (end_time - start_time)
print(f"Loading time: {execution_time:.5f} seconds")

# Tokenize the text using Baidu LAC
text = "三星T炫纹平板白"
#text = remove_non_alphabet_characters(text)
results = lac.run(text)
tokens = results[0]
print(tokens)

# Calculate the vector representation of the text using Tencent word vectors
start_time = time.time()
text_vector = z_score_normalization(np.mean([tencent_word_vectors.get(word, np.zeros(200)) for word in tokens], axis=0))
end_time = time.time()
execution_time = (end_time - start_time)
# 'text_vector' now contains the vector representation of the text
print(text_vector)
print(f"Execution time: {execution_time:.5f} seconds")
