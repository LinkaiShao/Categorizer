import re

text = "三星note 8客盟盔甲红"

# Create a translation table to remove numbers and non-word characters
translation_table = str.maketrans('', '', '1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

# Apply the translation to the text
cleaned_text = text.translate(translation_table)
cleaned_text = re.sub(r'[^\w\sぁ-んァ-ヶ一-龯]', '', cleaned_text)

print(cleaned_text)
