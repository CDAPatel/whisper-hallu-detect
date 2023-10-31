"""
Title: Utility Script for Hallucination Detector
Author: Carlin Patel, z5259674
Description: 
    A script that contains helper functions used in hallu_detect.py

Overview of important functions:
    - clean_string(): Formats input transcription
    - find_gradient(): Finds the gradient of token path
    - gaussian_smooth(): Applys a gaussian window to smooth the calculated gradient
    - peak_detection(): Finds peaks in the smoothed gradient & determines if there is a hallucination
"""

import string
import inflect
import re
import numpy as np
from scipy.signal import windows, convolve, find_peaks
from dataclasses import dataclass
from typing import List, Dict

def clean_string(text):
    # Check for invalid characters. Detector is only built for English
    flag = detect_chars(text)
    if flag:    # If invalid characters are detected, then the transcription is an error
        return text, flag
    
    # Change common honorifics into full words
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
    for i, word in enumerate(words):
        if "-" in word or "'" in word: # Only punctuation left is " - " and " ' ", so add a special case
            splits = []
            tokens = []

            # Split the word where - or ' occur
            prev_idx = 0
            for j, char in enumerate(word):
                if char in ["-","'"]:
                    tokens.append(word[prev_idx:j])
                    splits.append(char)
                    prev_idx = j+1
            tokens.append(word[prev_idx:])  # make sure to include the last segment

            # Convert tokens that are numbers
            tokens = [token_to_num(token) for token in tokens]

            # Reconstruct with the original punctuation
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

    if text.isdigit(): # Numbers only
        if 1000 <= int(text) <= 9999: # Special case for handling numbers that are read as years
            first = p.number_to_words(text[:2])
            second = p.number_to_words(text[2:])
            return f"{first} {second}" 
        else:   # All other numbers
            return p.number_to_words(text)
    elif text[:-2].isdigit() and text[-2:] in ['st', 'nd', 'rd', 'th']: # Ordinals
        return p.number_to_words(p.ordinal(text[:-2]))
    elif contains_digits(text): # Numbers adjacent to text
        sequences = re.findall(r'\d+', text)
        for seq in sequences:
            number = p.number_to_words(seq)
            if 1000 <= int(seq) <= 9999: # same case handling as above
                first = p.number_to_words(text[:2])
                second = p.number_to_words(text[2:])
                number = f"{first} {second}"   
            text = text.replace(seq, number, 1)
        return text
    else:   # If no numbers, just return the original text
        return text

def contains_digits(word):
    return any(char.isdigit() for char in word)

def find_gradient(values, window_size):
    # Calculate gradient using a given window size
    result = []
    for i in range(len(values)):
        if i < window_size or i > len(values) - window_size - 1: # Unsure if this is the correct descision.
            result.append(0)
            continue
        gradient = (values[i + window_size - 1] - values[i])/ (window_size - 1)
        result.append(gradient)
        
    return result

# TODO: Determine if this function needs to be removed
def gaussian_smooth(values, window_size, std):
    # Create a Gaussian window
    window = windows.gaussian(window_size, std)  
    return convolve(values, window/window.sum(), mode='same')

def moving_average(values, window_size):
    return convolve(values, np.ones(window_size)/window_size, mode='same')

def peak_detection(data, mean_threshold, peak_distance, height_threshold):
    # Find indexs of peaks that meet mean_threshold requirement
    peaks, _ = find_peaks(data, height=height_threshold, distance=peak_distance)

    # Result will hold all segments that have been identified as potential hallucinations
    result = []

    for peak in peaks:
        # Determine size of peak
        start = max(0, peak - peak_distance//2)
        end = min(len(data), peak + peak_distance//2)
        seg = data[start:end] # think this might need adjusting?
        seg_mean = np.mean(seg)
        seg_std = np.std(seg)

        # Check if peak meets conditions to be classified as a hallucination
        if seg_mean >= mean_threshold:
            result.append((start, end))

    return result

@dataclass
class Point:
    token_index: int
    time_index: int