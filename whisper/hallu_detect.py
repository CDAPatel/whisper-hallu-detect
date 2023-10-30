import torch
import torchaudio
import numpy as np

from .hallu_detect_utils import clean_string, find_gradient, gaussian_smooth, peak_detection, Point

# Do i want to define the hyper params in the hallu_detect call? or here.

def hallu_detect(
    transcript=None,
    audio=None,
    grad_window=5,
    gauss_window=5,
    gauss_std=2,
    mean_threshold=0.85,
    std_threshold=0.05,
    peak_distance=4,
    is_test=False # Do i need this?
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # First check for invalid input and that CUDA is being used
    
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

    if error_flag:
        print("Errors detected in transcript.")
        result = 'Error'
        return result

    # Then generate emission matrix
    emission, labels, waveform = generate_emission(audio, device)
    # waveform currently included as a return for plotting later 

    # Tokenise transcript
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in clean_transcript]

    # generate trellis
    trellis = generate_trellis(emission, tokens)

    # Find the path
    path = path_finding(trellis, emission, tokens)

    # Extract values
    values = []
    for point in path:
        values.append(point.token_index)
    
    # Gradient
    gradient = find_gradient(values, grad_window)

    # smoothing
    gradient_smooth = gaussian_smooth(gradient, gauss_window, gauss_std)
    
    # peak detection & threshold check
    segments = peak_detection(gradient_smooth, mean_threshold, std_threshold, peak_distance)
    
    if segments:
        print("Hallucination detected in transcript.")
        result = 'Hallucination'

    return result
    #TODO: More work required for word based detection. 

def generate_emission(audio, device):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()    
    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio)
        emissions, _ = model(waveform.to(device)) 
        emissions = torch.log_softmax(emissions, dim=-1)
    # The emission matrix produces a probability for each label at each frame
    emission = emissions[0].cpu().detach()

    return emission, labels, waveform

def generate_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens) # We assume that all the tokens in the transcript are present

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)) # 170x46
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0) # The bottom row, probabilities from emission matrix, k0 = 1
    trellis[0, -num_tokens:] = -float("inf")
    #trellis[-num_tokens:, 0] = float("inf") # Should I delete this????

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

def path_finding(trellis, emission, tokens, blank_id=0):


    # back to trying FORCED alginment
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