import string
import inflect
import re
import numpy as np
from scipy.signal import windows, convolve, find_peaks
from dataclasses import dataclass
from typing import List, Dict

def clean_string(text):
    # Need to:
    #   - Make all uppercase
    #   - Remove all punctuation
    #   - Replace whitespaces with '|'
    #   - Convert numbers to text
    
    # Initialise inflect, used to convert numbers
    

    # flag is used to detect errors
    # first we check if there are any non-allowed characters. This detector is only built for english, a True return indicates an error.
    flag = detect_chars(text)
    if flag:
        return text, flag
    
    # Alter titles
    text = replace_titles(text)
    
    # Remove punctuation except - and '
    exclude = set(string.punctuation) - {"'", "-"}
    text =''.join(ch if ch not in exclude else ' ' for ch in text)

    # Convert numbers to words
    text = convert_numerals(text)

    # have to clean again, this is not an elegant solution
    # TODO: Can I fixed this by changing the and in number_to_words?
    clean_text =''.join(ch if ch not in exclude else '' for ch in text)

    # Convert to uppercase and replace spaces with |
    clean_text.lstrip(" ")
    clean_text = clean_text.upper()
    clean_text = clean_text.replace(' ', '|')

    return clean_text, flag

def detect_chars(text):
    valid_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")

    for c in text:
        if c not in valid_chars:
            return True
    return False

def convert_numerals(text):
    words = text.split()

    # Iterate through each word and look for numbers.
    # Only punctuation characters left are ' and -, so add a special case for them

    for i, word in enumerate(words):
        # Check for - or '
        if "-" in word or "'" in word:
            splits = []
            tokens = []

            # Split the word by ' or -
            prev_idx = 0
            for j, char in enumerate(word):
                if char in ["-","'"]:
                    tokens.append(word[prev_idx:j])
                    splits.append(char)
                    prev_idx = j+1
            tokens.append(word[prev_idx:])  # make sure to include the last segment

            # Now convert tokens that are numbers
            tokens = [token_to_num(token) for token in tokens]

            # Reconstruct with the splitters
            new_word = tokens[0]

            for j, split in enumerate(splits):
                new_word += split + tokens[j+1]
            words[i] = new_word
        else:
            words[i] = token_to_num(word)
    
    return ' '.join(words)

def replace_titles(text):
    # Dictionary of honorifics. Not complete.
    title_dict = {
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miss",
        "Dr.": "Doctor",
        "Prof.": "Professor",
        "Rev.": "Reverend",
        "Sr.": "Senior",
        "Jr.": "Junior",
        "Hon.": "Honourable",
        "Capt.": "Captain" 
    } 

    for short, full in title_dict.items():
        text = text.replace(short, full)

    return text

def token_to_num(text):
    p = inflect.engine()

    if text.isdigit(): # Only a number
        if 1000 <= int(text) <= 9999: # Special case for handling dates
            first = p.number_to_words(text[:2])
            second = p.number_to_words(text[2:])
            return f"{first} {second}" 
        else:   # all other numbers
            return p.number_to_words(text)
    elif text[:-2].isdigit() and text[-2:] in ['st', 'nd', 'rd', 'th']: # ordinals
        return p.number_to_words(p.ordinal(text[:-2]))
    elif contains_digits(text):
        sequences = re.findall(r'\d+', text)
        for seq in sequences:
            number = p.number_to_words(seq)
            if 1000 <= int(seq) <= 9999: # Special case for handling dates
                first = p.number_to_words(text[:2])
                second = p.number_to_words(text[2:])
                number = f"{first} {second}"   
            text = text.replace(seq, number, 1)
        return text
    else:
        return text

def contains_digits(word):
    return any(char.isdigit() for char in word)

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

def peak_detection(data, mean_threshold, std_threshold, peak_distance):
    peaks, _ = find_peaks(data, height=mean_threshold, distance=peak_distance)

    # Result will hold all segments that have been identified as potential hallucinations.
    result = []

    for peak in peaks:
        start = max(0, peak - peak_distance//2)
        end = min(len(data), peak + peak_distance//2)
        seg = data[start:end] # think this might need adjusting?
        seg_mean = np.mean(seg)
        seg_std = np.std(seg)

        if seg_mean >= mean_threshold and seg_std <= std_threshold:
            result.append((start, end))

    return result

@dataclass
class Point:
    token_index: int
    time_index: int