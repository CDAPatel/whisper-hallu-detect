import string
import inflect
from scipy.signal import windows, convolve
from dataclasses import dataclass
from typing import List, Dict

def clean_string(text):
    # Need to:
    #   - Make all uppercase
    #   - Remove all punctuation
    #   - Replace whitespaces with '|'
    #   - Convert numbers to text
    
    # Initialise inflect, used to convert numbers
    p = inflect.engine()

    # Remove punctuation except - and '
    exclude = set(string.punctuation) - {"'", "-"}
    text =''.join(ch if ch not in exclude else ' ' for ch in text)

    # Convert numbers to words
    #TODO: WHen there is a word that is '1-1', the dash is not removed nor is the word recognised
    # as a number.
    words = text.split()
    clean_words = []
    for word in words:
        if word.isdigit():
            word = p.number_to_words(word)
        clean_words.append(word)

    clean_text = ' '.join(clean_words)

    # Convert to uppercase
    clean_text = clean_text.upper()

    # Replace spaces with |
    clean_text = clean_text.replace(' ', '|')

    return clean_text

def find_gradient(values, window_size):
    result = []
    for i in range(len(values)):
        if i < window_size or i > len(values) - window_size - 1: # Unsure if this is the correct descision.
            result.append(0)
            continue
        gradient = (values[i + window_size - 1] - values[i])/ (window_size - 1)
        result.append(gradient)
        
    return result

def gaussian_smooth(values, window_size, std):
    # Create a Gaussian window
    window = windows.gaussian(window_size, std)  
    return convolve(values, window/window.sum(), mode='same')

@dataclass
class Point:
    token_index: int
    time_index: int