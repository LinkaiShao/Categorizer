import numpy as np
from langdetect import detect

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
# -1 other language

def determine_language(text):
    language_mapping_lang_detect = {
    'en': 0,
    'ja': 1,
    'zh-cn': 2,
    'es': 3
    }
    lang = 0
    # chinese might have weird eng in it
    for char in text:
        rep = repetition
        if(is_hiragana_or_katakana(char)):
            return 1
        if(is_simplified_or_traditional_chinese(char)):
            lang = 2
    if(lang != 1 and lang != 2):
        try:
            r = detect(text)
            lang = language_mapping_lang_detect.get(r, -1)
        except Exception as e:
            return -1
            
        
    print(lang)
    return lang
repetition = 3 # lang detect repetition
def detect_languages_in_text_file(file_path):
    language_counts = {0: 0, 1: 0, 2: 0 , 3:0}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            
            item = line.strip()  # Remove leading/trailing whitespaces
            print(item)
            for _ in range(3):
                try:
                    language_code = determine_language(item)
                    language_counts[language_code] += 1
                    break
                except:
                    pass  # Skip the line

    return language_counts

def detect_languages_in_text_file2(file_path):
    language_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    other_languages = {}  # Dictionary to store other frequent languages

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = line.strip()  # Remove leading/trailing whitespaces
            language_code = determine_language(item)
            if language_code in language_counts:
                language_counts[language_code] += 1
            else:
                if language_code in other_languages:
                    other_languages[language_code] += 1
                else:
                    other_languages[language_code] = 1
    sorted_other_languages = dict(
        sorted(other_languages.items(), key=lambda item: item[1], reverse=True)
    )

    return language_counts, sorted_other_languages

# Example usage:
file_path = 'C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\36k5\\36k5-copy.csv'  # Replace with the path to your text file
result,others = detect_languages_in_text_file2(file_path)
print("Language Code 0 (English):", result[0])
print("Language Code 1 (Japanese):", result[1])
print("Language Code 2 (Chinese):", result[2])
print("Language Code 3 (Spanish):", result[3])
print(result[0] + result[1] + result[2])
print("Other Frequent Languages:", others)