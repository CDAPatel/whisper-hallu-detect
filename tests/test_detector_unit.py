"""
 Script that is used to test the hallucination detector on individual cases.
"""

from whisper.hallu_detect import hallu_detect

def main():
    # Hard define the transcript and audio
    # Will be locally defined for my machine.
    transcript_in = " SHE RAN TO HER HUSBAND'S SIDE AT ONCE AND HAD... SHE HAD TO RUN TO HER HUSBAND'S SIDE AT ONCE AND HAD... HELPS HIM LIFT THE FOUR CATTLES FROM THE FIRE."
    audio_in = "C:\dev\librispeech-pause\perturb_audio\\1284-1181-0009_50pc_25s.flac"


    # Simple call to hallu_detect()
    result = hallu_detect(transcript=transcript_in, audio=audio_in, word_detect=True)

    # Correct transcript of above
    print("SHE RAN TO HER HUSBAND'S SIDE AT ONCE AND HELPED HIM LIFT THE FOUR KETTLES FROM THE FIRE")

if __name__ == "__main__":
    main()