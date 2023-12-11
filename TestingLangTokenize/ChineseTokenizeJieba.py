import jieba
import time
def load_tencent_word_vectors(file_path):
    # Initialize an empty dictionary to store word vectors
    word_vectors = {}
    
    # Open and read the Tencent word embedding text file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line into word and vector parts
            parts = line.strip().split(' ')
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            
            # Store the word vector in the dictionary
            word_vectors[word] = vector
    
    return word_vectors

# Load Tencent word vectors
tencent_word_vectors = load_tencent_word_vectors('C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\WordEmbedding\\tencentAi\\tencentAi\\tencent_embedding.txt')

def description_to_vector(description, word_vectors, default_vector=None):
    start_time = time.time()
    # Tokenize the input description using jieba for Chinese text
    words = jieba.lcut(description)
    end_time = time.time()
    execution_time = start_time - end_time
    print(f"Jieba execution time: {execution_time:.5f} seconds")
    # Initialize an empty vector
    vector = default_vector if default_vector is not None else [0.0] * len(word_vectors.get(list(word_vectors.keys())[0]))
    
    # Count the number of words for later normalization
    word_count = 0
    start_time = time.time()
    # Calculate the vector for the description
    for word in words:
        if word in word_vectors:
            vector += word_vectors[word]
            word_count += 1
    
    # Normalize the vector by dividing by the number of words (optional)
    if word_count > 0:
        vector = [val / word_count for val in vector]
    end_time = time.time()
    execution_time = start_time - end_time
    print(f"Vector execution time: {execution_time:.5f} seconds")
    
    return vector

# Example usage:
description = "三星note 8客盟盔甲红"
start_time = time.time()
vector = description_to_vector(description, tencent_word_vectors)
end_time = time.time()
execution_time = end_time - start_time
# You can use 'vector' for further analysis or tasks
print(vector)
print(f"Execution time: {execution_time:.5f} seconds")
