import wordsegment

wordsegment.load()

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

# Example usage
text = "Lavieestcommeunebicyclette,pourgarderl'équilibre,ilfautavancer."
special_characters = [
    'é', 'è', 'ê', 'ë', 'à', 'â', 'î', 'ï', 'ô', 'û', 'ü',
    'æ', 'œ', 'ç'
]
segmented_words = wordsegment.segment(text)

corrected_words = correct_segmented_words(segmented_words, text)
print(corrected_words)

