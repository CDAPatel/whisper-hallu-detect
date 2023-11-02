"""
    Script that is used to test the hallucination detector on a large pre-annotated dataset.
"""

import pandas as pd
import os
import torch
import torchaudio
from whisper.hallu_detect import hallu_detect

data_path = "C:\\dev\\librispeech-pause\\perturbed_dataset_annotated_clean.csv"
audio_path = "C:\\dev\\librispeech-pause\\perturb_audio"
result_path = "C:\\dev\\whisper-hallu-detect\\test_results\\full_tests\\threshold_85_window_5_dist_2.csv"

def main():
    # Need to load the data
    data = pd.read_csv(data_path)
    # iterate through and send to the detector

    # Initialise wav2vec here to save time.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    for idx, row in data.iterrows():
        #if idx > 1000: 
        #    break

        transcription = row['transcription']
        filename = row['filename']
        audio_file = os.path.join(audio_path, filename)

        print("Testing index: ", idx)

        # Call detector
        result = hallu_detect(transcript=transcription, audio=audio_file, window_size=5, seg_threshold=3, model=model, device=device, bundle=bundle, is_test=True)

        # result handling and add to data
        if result == 'Detector Error':
            value = 'D'
        elif result == 'None':
            value = 'N'
        elif result == "Hallucination":
            value = 'H'
        else:
            value = 'E'

        data.at[idx, 'detector'] = value

        if idx % 50 == 0:
            data.to_csv(result_path, index=False)
            print('file saved...')   

    data.to_csv(result_path, index=False)
    print('Testing complete')

if __name__ == "__main__":
    main()