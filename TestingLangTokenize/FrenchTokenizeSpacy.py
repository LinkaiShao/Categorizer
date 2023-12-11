import re

# Example French text
french_text = "C'estunephraseenfrançais."

# Define a regular expression pattern to identify French words with contractions
pattern = r"\b\w+\b"

# Tokenize the text using the regular expression pattern
tokens = re.findall(pattern, french_text, re.IGNORECASE)

# Print the individual tokens
for token in tokens:
    print(token)
