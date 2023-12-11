import numpy as np
import re
import time
import spacy
import nltk
import wordninja
def z_score_normalization(embedding):
    mean = np.mean(embedding)
    std_dev = np.std(embedding)
    standardized_embedding = (embedding - mean) / std_dev
    return standardized_embedding

def remove_non_alphabet_characters(input_string):
    # Use a regular expression to remove all non-alphabet characters
    cleaned_string = re.sub(r'[^a-zA-Z ]', '', input_string)
    return cleaned_string

# Load GloVe vectors (change the path to your glove.6B.200d.txt file)
def load_glove_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line into word and vector parts
            parts = line.strip().split(' ')
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]], dtype='float32')

            # Store the word vector in the dictionary
            word_vectors[word] = vector

    return word_vectors

def custom_tokenizer(nlp):
    return spacy.tokenizer.Tokenizer(nlp.vocab, rules={"split_non_space": [{"ORTH": r"\S+"}]})

# Load GloVe word vectors

start_time = time.time()
glove_file = 'C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\WordEmbedding\\glove.6B\\glove.6B.200d.txt'
word_vectors = load_glove_vectors(glove_file)
end_time = time.time()
execution_time = (end_time - start_time)*1
print(f"loading time: {execution_time:.5f} seconds")

def custom_tokenizer(text):
    tokens = []
    current_token = ""

    for char in text:
        if char.isalpha():
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
            current_token = ""

    if current_token:
        tokens.append(current_token)

    return tokens

def generate_english_vector(text, word_vectors):
    # Tokenize the text into words
    tokens = wordninja.split(text)
    

    # Print the original text and individual tokens
    print("Original text:")
    print(text)
    print("Individual tokens:")
    for token in tokens:
        print(token)

    # Calculate the vector representation of the text
    text_vector = np.mean([word_vectors.get(token, np.zeros(200)) for token in tokens], axis=0)
    return z_score_normalization(text_vector)
# Tokenize the text using nltk
#text = "hellomyname"
#text = remove_non_alphabet_characters(text)
#nltk.data.path.append("C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\punkt")
#words = nltk.word_tokenize(text)
nlp = spacy.load("en_core_web_sm")

# Your text with no spaces
text = "hello my name"

# Tokenize the text using spaCy
#words = nlp(text)

# tokenize the text using re
words = wordninja.split(text)
for token in words:
    print("Word Tokens:", token)
tokens = text.lower().split()  # You can convert text to lowercase and split it into words

# Calculate the vector representation of the text
start_time = time.time()
# average around the first
text_vector = generate_english_vector(text, word_vectors)
end_time = time.time()
execution_time = (end_time - start_time)*1
# 'text_vector' now contains the vector representation of the text
print(text_vector)
print(f"Execution time: {execution_time:.5f} seconds")