from langdetect import detect
from collections import Counter

def detect_languages_in_text_file(file_path):
    language_counts = Counter()  # Initialize a counter to count language occurrences
    language_counts["not a language"] = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = line.strip()  # Remove leading/trailing whitespaces
            try:
                detected_language = detect(item)
                language_counts[detected_language] += 1
            except:
                language_counts["not a language"] += 1
                pass  # Skip items that couldn't be detected

    # Sort the detected languages by count (from most to least)
    sorted_languages = language_counts.most_common()

    return sorted_languages

# Example usage:
file_path = 'C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\36k5\\36k5-copy.csv'  # Replace with the path to your text file
sorted_languages = detect_languages_in_text_file(file_path)

print("Most prevalent languages based on langdetect:")
for lang, count in sorted_languages:
    print(f"{lang}: {count}")
