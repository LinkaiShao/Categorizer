import numpy as np

def one_hot_encode(description, charset):
    # Initialize a binary vector with zeros
    one_hot_vector = np.zeros(len(charset), dtype=int)
    
    # Iterate through the characters in the description
    for char in description:
        if char in charset:
            # Find the index of the character in the charset
            index = charset.index(char)
            # Set the corresponding element in the one-hot vector to 1
            one_hot_vector[index] = 1
    
    return one_hot_vector

# Define the character set (assuming a simplified character set)
charset = '三星note8客盟盔甲红'

# Example usage:
description = "三星note 8客盟盔甲红"
one_hot = one_hot_encode(description, charset)

# The 'one_hot' variable contains the one-hot encoded vector
print(one_hot)
