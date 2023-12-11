import MeCab
import numpy as np
import time

# Initialize MeCab
m = MeCab.Tagger()
def z_score_normalization(embedding):
    mean = np.mean(embedding)
    std_dev = np.std(embedding)
    standardized_embedding = (embedding - mean) / std_dev
    return standardized_embedding
# Load Japanese word vectors (depja200)
def depja200_word_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

# Load Tencent Japanese word vectors
tencent_word_vectors = depja200_word_vectors('C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\WordEmbedding\\dep-ja-200dim.txt')  # Replace with the actual file path

# Function to tokenize and generate text vector
def generate_japanese_vector(text, word_vectors, mecab):
    # Tokenize the text into words
    node = mecab.parse(text)
    tokens = [line.split('\t')[0] for line in node.split('\n') if line]

    # Print the original text and individual tokens
    print("Original text:")
    print(text)
    print("Individual tokens:")
    for token in tokens:
        print(token)

    # Calculate the vector representation of the text
    text_vector = np.mean([word_vectors.get(token, np.zeros(200)) for token in tokens], axis=0)
    return z_score_normalization(text_vector)

# Example usage:
start_time = time.time()
text = "三面鏡360°セルフカット鏡化粧鏡折りたたみ式3面鏡三面鏡壁掛け3ウェイミラーHD鏡面角度調整高さ調節可安定して使い飛散防止卓上でも使用"
vector = generate_japanese_vector(text, tencent_word_vectors, m)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.5f} seconds")
print(vector)
