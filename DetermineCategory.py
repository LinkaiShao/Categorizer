import numpy as np
from LAC import LAC
import time
import re
import MeCab
import spacy
import wordninja
from langdetect import detect
import wordsegment
import tensorflow as tf
from tensorflow.keras.models import load_model



pretrained_base_path = 'D:\\WordEmbedding300'
cn300_embedding_path = "D:\\WordEmbedding300\\cc.zh.300.vec\\cc.zh.300.vec"
jp300_embedding_path = "D:\\WordEmbedding300\\cc.ja.300.vec\\cc.ja.300.vec"
en300_embedding_path = "D:\\WordEmbedding300\\cc.en.300.vec\\cc.en.300.vec"
es300_embedding_path = "D:\\WordEmbedding300\\cc.es.300.vec\\cc.es.300.vec"
fr300_embedding_path = "D:\\WordEmbedding300\\cc.fr.300.vec\\cc.fr.300.vec"
de300_embedding_Path = "D:\\WordEmbedding300\\cc.de.300.vec\\cc.de.300.vec"
language_mapping_lang_detect = {
    'en': 0,
    'ja': 1,
    'zh-cn': 2,
    'es': 3,
    'it': 3,
    'fr': 4,
    'de': 5
 }
# count the amount of languages we scanned
language_counts = {0: 0, 1: 0, 2: 0, 3:0, 4:0, 5:0, -1:0}


one_hot_categories = {
    "clothing/bags": [1, 0, 0, 0, 0, 0],
    "medicine/food": [0, 1, 0, 0, 0, 0],
    "makeup": [0, 0, 1, 0, 0, 0],
    "electronics": [0, 0, 0, 1, 0, 0],
    "furniture": [0, 0, 0, 0, 1, 0],
    "others": [0, 0, 0, 0, 0, 1]
}

##################################################################################################
####### token splitters 
# Initialize Baidu LAC
lac = LAC(mode='rank')
def split_tokens_cn_lac(text):
    results = lac.run(text)
    tokens = results[0]
    return tokens

def split_tokens_eng_wordninja(text):
    tokens = wordninja.split(text)
    return tokens

def split_tokens_jp_mecab(text):
    m = MeCab.Tagger()
    node = m.parse(text)
    tokens = [line.split('\t')[0] for line in node.split('\n') if line]
    return tokens
wordsegment.load()
def split_tokens_es_word_segment(text):
    return wordsegment.segment(text)

def correct_segmented_words(segmented_words, text):
    text = text.lower()
    orig_itr = 0
    orig_len = len(text)
    seg_itr = 0
    seg_w_itr = 0
    
    words = []
    cur_word = ""
    
    while orig_itr < orig_len and seg_itr < len(segmented_words):
        s = segmented_words[seg_itr][seg_w_itr]
        t = text[orig_itr]
        if s == t:
            cur_word += t
            seg_w_itr += 1
            
            if seg_w_itr == len(segmented_words[seg_itr]):  # move to the next word
                seg_itr += 1
                seg_w_itr = 0
                words.append(cur_word)
                cur_word = ""
        else:
            cur_word += t  # Use the character from text
            
        orig_itr += 1

    # Append the last word if any
    if cur_word:
        words.append(cur_word)

    return words

def split_tokens_french_custom(text):
    segmented_words = wordsegment.segment(text)
    tokens = correct_segmented_words(segmented_words, text)
    return tokens

def split_tokens_german_custom(text):
    segmented_words = wordsegment.segment(text)
    tokens = correct_segmented_words(segmented_words, text)
    return tokens

######## end of token splitters
###################################################################################################################
######## language determination functions
# hiragana and katakana = jp, kanji might be chinese
def is_hiragana_or_katakana(char):
    code_point = ord(char)
    return (0x3040 <= code_point <= 0x309F) or (0x30A0 <= code_point <= 0x30FF)

# traditional or simplified chinese = chinese letter, might be jp tho
def is_simplified_or_traditional_chinese(char):
    code_point = ord(char)
    # Range for Simplified Chinese (common): U+4E00 to U+9FFF
    simplified_range = (0x4E00 <= code_point <= 0x9FFF)
    # Range for Traditional Chinese (common): U+3400 to U+4DBF and U+20000 to U+2A6DF
    traditional_range = (0x3400 <= code_point <= 0x4DBF) or (0x20000 <= code_point <= 0x2A6DF)
    return simplified_range or traditional_range

# 0 eng
# 1 jp
# 2 cn
# 3 es
# 3 it
# italian and spanish have the same root, which is latin
# 4 fr
# 5 de
# -1 other language
def determine_langauge(text):
    lang = 0
    # chinese might have weird eng in it
    for char in text:
        if(is_hiragana_or_katakana(char)):
            return 1
        if(is_simplified_or_traditional_chinese(char)):
            lang = 2
    if(lang != 1 and lang != 2):
        # lang detect might fail due to no language given

        try:
            lang2 = detect(text)
            
            print(f"lang detect detected language: {lang2}")
            lang = language_mapping_lang_detect.get(lang2, -1)
        except Exception as e:
            return -1
    return lang
######## end of language determination
#############################################################################################################
######## text cleaner
# rid numbers
def clean_text1(text):
    text = text.strip()
    # Create a translation table to remove numbers and non-word characters
    translation_table = str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    #translation_table = str.maketrans('', '', '1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    # Apply the translation to the text
    cleaned_text = text.translate(translation_table)
    cleaned_text = re.sub(r'[^\w\sぁ-んァ-ヶ一-龯]', '', cleaned_text)
    return cleaned_text

# dont rid numbers
def clean_text2(text):
    text = text.strip()
    # Create a translation table to remove numbers and non-word characters
    #translation_table = str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    translation_table = str.maketrans('', '', '1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    # Apply the translation to the text
    cleaned_text = text.translate(translation_table)
    cleaned_text = re.sub(r'[^\w\sぁ-んァ-ヶ一-龯]', '', cleaned_text)
    return cleaned_text
######### end of text cleaner
########################################################################################################
language_mappings = {
        1: ("jp sentence", split_tokens_jp_mecab, jp300_embedding_path),
        2: ("cn sentence", split_tokens_cn_lac, cn300_embedding_path),
        0: ("eng sentence", split_tokens_eng_wordninja, en300_embedding_path),
        3: ("latin sentence", split_tokens_es_word_segment, es300_embedding_path),
        4:("french sentence", split_tokens_french_custom,fr300_embedding_path),
        5:("german sentence",split_tokens_german_custom , de300_embedding_Path)
    }
#######################################################################################################
########## split the input and categories by langauge 
def split_by_language(input):
    # splitting the input into different languages
    language_inputs = {
        -1:[],
        0 :[],
        1: [],
        2:[],
        3:[],
        4:[],
        5:[]
    }
    counter = 0
    for item in input:
        lang = determine_langauge(item)
        language_inputs[lang].append((item.strip(), counter))
        counter += 1
    return language_inputs
###############################################################################################item########
########## loading and calculation of word vectors, vector calculations
def load_word_vectors(file_path):
    print("loading word embedding vectors")
    print(file_path)
    start_time = time.time()
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
    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Loading time: {execution_time:.5f} seconds")
    return word_vectors

def calculate_text_vector(tokens, word_vectors):
    start_time = time.time()
    # if cant find, we do 200 0s, or else we add it on there and average
    text_vector = np.mean([word_vectors.get(word.lower(), np.zeros(300)) for word in tokens], axis=0)
    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Vector Generation Execution time: {execution_time:.5f} seconds")
    return text_vector

# this is for generalizing the vectors generated by 3 different word embedding files
def z_score_normalization(embedding):
    mean = np.mean(embedding)
    std_dev = np.std(embedding)
    standardized_embedding = (embedding - mean) / std_dev
    return standardized_embedding
#######################################################################################################
############print results
def print_language_done():
    for code, count in language_counts.items():
        if code == -1:
            print(f"Language Code {code} (Tokenize Failed): {count}")
        elif code in language_mapping_lang_detect.values():
            language_codes = [key for key, value in language_mapping_lang_detect.items() if value == code]
            for language_code in language_codes:
                print(f"Language Code {code} ({language_code}): {count}")
    total_count = sum(language_counts.values())
    print("Total Count:", total_count)  

def sort_categories(input):
    sorted_categories = sorted(input, key=lambda x: x[1])
    # Extract only the predictions (categories)
    sorted_predictions = [prediction for prediction, _ in sorted_categories]
    return sorted_predictions

#######################################################################################################
########## make prediction
def predict_category(model, input_vector, category_mapping):
    # Reshape the input_vector to match the model's input shape
    input_vector = np.array(input_vector).reshape(1, -1)
    
    # Get the model's prediction
    prediction = model.predict(input_vector)
    print(prediction)
    # Get the index of the maximum probability
    predicted_index = np.argmax(prediction, axis=1)[0]
    
    # Convert the index to word
    predicted_category = list(category_mapping.keys())[predicted_index]
    
    return predicted_category
#######################################################################################################
def vectorize_and_categorize(language_inputs, model_path):
    model = load_model(model_path)
    all_categories = []
    for lang, lang_data in language_inputs.items():
        if lang in language_mappings:
            cur_lang = language_mappings[lang]
            sentence_type, tokenization_function, embedding_path = cur_lang
            cur_vectors = load_word_vectors(embedding_path)
            for description, counter in lang_data:
                print(sentence_type)
                vectors = np.zeros(300)
                # run clean text1
                cleaned1 = clean_text1(description)
                # tokenize
                tokens = tokenization_function(cleaned1)
                for token in tokens:
                    print(token)
                vectors = calculate_text_vector(tokens, cur_vectors)
                if np.all(np.isnan(vectors) | (vectors == 0)):
                    cleaned2 = clean_text2(description)
                    tokens = tokenization_function(cleaned2)
                    vectors = calculate_text_vector(tokens,cur_vectors)
                    for token in tokens:
                        print(token)
                    if np.all(np.isnan(vectors) | (vectors == 0)):
                        # undoable, skipping
                        print("unable to vectorize, putting others")
                        all_categories.append(("others",counter))
                        language_counts[-1] += 1
                        continue
                # we have the vectors
                print("pre zscore:")
                print(vectors)
                res = z_score_normalization(vectors)
                print("post zscore:")
                print(res)
                # put it into the grinder
                prediction = predict_category(model,res,one_hot_categories)
                all_categories.append((prediction, counter))
                language_counts[lang] += 1
            del cur_vectors
        else:
            #language = -1
             for description, counter in lang_data:
                 all_categories.append(("others",counter))
    r = sort_categories(all_categories) 
    # tally 
    print_language_done()
    return r
    
#################################################################################################################

                
file_path = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\36k2\\36k2-copy.csv"  # Replace with the actual path to your file
model_path = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\WordEmbedding\\TrainedNN\\double_dropout300.h5"
output_path = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\36k2\\predictions.txt"
 #Read lines from the file
with open(file_path, "r", encoding="utf-8") as file:
   lines = file.readlines()
lang_inputs = split_by_language(lines)
result = vectorize_and_categorize(lang_inputs, model_path)
with open(output_path, "a") as file:
    for line in result:
      file.write(line + "\n")
def run_categorizer(input):
    model_path = "C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\WordEmbedding\\TrainedNN\\double_dropout300.h5"
    lang_inputs = split_by_language(input)
    result = vectorize_and_categorize(lang_inputs, model_path)
    return result



