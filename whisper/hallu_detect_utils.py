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
from dataclasses import dataclass
from typing import List, Dict
from colorama import init, Fore, Style
init(autoreset=True)

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

def segment_detection(data, seg_threshold):
    # Finds all segments of consecutive 1s that are of length seq_threshold or larger
    # Returns the start and end indexes of these segments
    curr_length = 0
    seg_start_idx = None

    result = []

    for i, val in enumerate(data):
        # Check if the current element is 1
        if val == 1:
            # Start a new sequence if the current sequence length is 0
            if curr_length == 0:
                seg_start_idx = i
            # Increment the sequence length
            curr_length += 1
        else:
            # Check length of segment against threshold
            if curr_length >= seg_threshold:
                result.append((seg_start_idx, i-1))
            # Reset the current sequence length
            curr_length = 0
    
    # Check for a sequence at the end of the array
    if curr_length >= seg_threshold:
        result.append((seg_start_idx, len(data) - 1))
    
    return result

def segment_connection(segments, window_size):
    # Connects segments that are within window_size of each other together

    result = [segments[0]]

    for curr_start, curr_end in segments[1:]:
        prev_start, prev_end = result[-1]

        if curr_start - prev_end <= window_size:  # Two segments are close enough
            result[-1] = (prev_start, curr_end)
        else: # Add a new segment to the result
            result.append((curr_start, curr_end))

    return result

def expand_segments(segments, window_size, max_len):
    # Expands each segment by window_size in either direction
    # Accounts for the fact that we want to use the segment indices for the values not gradients
    result = []

    for start, end in segments:
        start = max(0, start - window_size)
        end = min(end + window_size, max_len)
        result.append((start, end))

    return result

def find_tokens(segments, values):
    # Convert the time index in the segments to the corresponding token index from the values
    result = []

    for start, end in segments:
        result.append((values[start], values[end]))

    return result

def print_hallus(transcript, segments):
    # Changes colour of suspected hallucinations to red and then prints

    # Replace | with whitespaces in transcript
    clean_transcript = transcript.replace('|', ' ')
    
    idx = 0
    result = ""

    for start, end in segments:
        result += clean_transcript[idx:start]
        result += Fore.RED + clean_transcript[start:end] + Fore.RESET
        idx = end
    
    result += clean_transcript[idx:]

    print(result)

@dataclass
class Point:
    token_index: int
    time_index: int