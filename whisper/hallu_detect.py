"""
Title: Hallucination Detector for Whisper
Author: Carlin Patel, z5259674
Description: 
    A script that takes an audio file and the Whisper produced transcript and attempts to detect
    hallucinations using forced alignment.

Inputs:
    transcript [string]: transcript produced by Whisper
    audio [string]: file path to corresponding audio

Outputs:
    Result [string]: which can take on these values:
    - "Hallucination" - indicates that a hallucination has been detected
    - "None" - no hallucination detected, could be a success or error
    - "Error" - invalid character detected in the transcript
    - "Detector Error" -  missing audio or transcript

Notes:
    - This project was created as part of an Elec. Eng. Thesis at the University of New South Wales
    - Large parts of this algorithm are adapted from:
        https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    - Whisper github:
        https://github.com/openai/whisper
"""

import torch
import torchaudio
from .hallu_detect_utils import clean_string, find_gradient, segment_detection, segment_connection, find_tokens, print_hallus, Point

def hallu_detect(
    transcript=None,
    audio=None,
    window_size=5,          # Size of gradient window
    seg_threshold=3,        # Minimum sequence length to be classified as a hallucination
    model = None,           # Temporary to reduce time while testing.
    device = None,
    bundle = None,
    word_detect = False,
    is_test= False           # Flag used for large scale testing so that the wav2vec model is not loaded every iteration
    #TODO: Add a flag for word detection
    ):
    # Find the device, check for invalid input
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if transcript is None:
        print("No transcription provided.")
        return 'Detector Error'
    if audio is None:
        print("No input audio provided.")
        return 'Detector Error'
    
    # Initialise default result
    result = 'None'

    # Clean transcript
    clean_transcript, error_flag = clean_string(transcript)

    if error_flag:  # Invalid character detected in the transcript
        print("Errors detected in transcript.")
        result = 'Error'
        return result

    # Generate the emission matrix - holds probabilites for each label at each (time) frame 
    if is_test:
        emission, labels, waveform = generate_emission(audio, device, model=model, bundle=bundle)
    else:
        emission, labels, waveform = generate_emission(audio, device)
    # TODO: waveform is only included for plotting functionality that will be added later

    # Tokenise transcript
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in clean_transcript]

    # Generate trellis
    trellis = generate_trellis(emission, tokens)

    # Find the most likely path by backtracking
    path = path_finding(trellis, emission, tokens)

    # Extract token index values for gradient calculation
    values = []
    for point in path:
        values.append(point.token_index)
    
    # Gradient calcuation
    gradient = find_gradient(values, window_size)

    # Find hallucinated segments
    segments = segment_detection(gradient, seg_threshold)
    
    if segments and not word_detect:    # If segments has elements then a hallucination has been detected
        print("Hallucination detected in transcript.")
        result = 'Hallucination'
    elif segments and word_detect: # Hallucination detected and word detection is requested
        print("Hallucination detected. Suspected hallucination highlighted in red")
        result = 'Hallucination'
        hallucinated_segments = segment_connection(segments, window_size, len(values))
        # Convert from time indexs to token indexs
        # Values is used as the indices in hallucinated_segments are originally calculated from values
        # This could be modified to use the path - but the fact that the path time index do not start from 0 needs to be accounted for
        hallucinated_tokens = find_tokens(hallucinated_segments, values)
        print_hallus(clean_transcript, hallucinated_tokens)

    return result
    #TODO: More work required for word based detection. 

def generate_emission(audio, device, model=None, bundle=None):
    if model is None:
        # Initialise the wav2vec model
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        
    labels = bundle.get_labels()
    # Generate the emission matrix
    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio)
        emissions, _ = model(waveform.to(device)) 
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()
    return emission, labels, waveform

def generate_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens) # Align all tokens

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0   # Initialise starting point for Trellis generation. Value is in log domain
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0) # Trellis values that correspond to never changing token
    # Block out invalid paths in the trellis
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf") 

    for t in range(num_frame):
        # Trellis is implemented one row at a time.
        # Rows -> Transcript
        # Columns -> Time
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            # Blank token corresponds to a repeat
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

def path_finding(trellis, emission, tokens, blank_id=0):
    # Take starting point as most probable point for final token in transcript
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change

        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]