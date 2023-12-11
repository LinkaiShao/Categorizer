from langdetect import detect

text = "Per Samsung Galaxy A21s Schermo di Ricambio Per Samsung A21s Schermo LCD A217 Display LCD SM-A217F Touch Digitizer Assembly SM-A217M, SM-A217N Kit di"
text2 = "Per Samsung Galaxy A21s Schermo di Ricambio Per Samsung A21s Schermo LCD A217 Display LCD SM-A217F Touch Digitizer Assembly SM-A217M, SM-A217N Kit di"
# Detect the language of the text
language = detect(text2)
language2 = detect(text)
# Print the detected language
print(f"Detected language: {language}")
print(f"Detected language: {language2}")
