def clear_file(file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.truncate(0)  # Truncate the file to remove all content
        print(f"Cleared the content of {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define the file paths to clear
file_paths = [
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\ChineseVectors.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\JapaneseVectors.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\EnglishVectors.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\LatinVectors.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\ChineseEndpoints.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\JapaneseEndpoints.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\EnglishEndpoints.txt',
    'C:\\Users\linka\\OneDrive\\Desktop\\Code\\HuiFu\\TrainingDataVectorized\\LatinEndpoints.txt'
]

# Clear the content of each file
for file_path in file_paths:
    clear_file(file_path)
