"""
 Script that is used to test the hallucination detector on individual cases.
"""

from whisper.hallu_detect import hallu_detect

def main():
    # Hard define the transcript and audio
    # Will be locally defined for my machine.
    transcript_in = "I WILL BRIEFLY DESCRIBE THEM TO YOU, AND YOU SHALL READ THE ACCOUNT OF THEM AT YOUR LINK IN THE DESCRIPTION. THANK YOU FOR YOUR LEISURE IN THE SACRED REGISTERS."
    audio_in = "C:\dev\librispeech-pause\perturb_audio\\2961-961-0016_75pc_20s.flac"


    # Simple call to hallu_detect()
    result = hallu_detect(transcript=transcript_in, audio=audio_in, word_detect=True)

    # Correct transcript of above
    print("Truth transcription:")
    print("I WILL BRIEFLY DESCRIBE THEM TO YOU AND YOU SHALL READ THE ACCOUNT OF THEM AT YOUR LEISURE IN THE SACRED REGISTERS")

if __name__ == "__main__":
    main()