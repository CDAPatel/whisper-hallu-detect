"""
 Placeholder description.
"""

from whisper.hallu_detect import hallu_detect

def main():
    # Hard define the transcript and audio
    # Will be locally defined for my machine.
    transcript_in = "Artificial intelligence has made significant strides and in the field of natural language processing. It allows machines to understand, generate, and respond to human language"
    audio_in = "C:\dev\Thesis\Forced Alignment\Initial Code\\audio\Exp_1_10_pausepos_7_pauselen_25s.mp3"


    # Simple call to hallu_detect()
    result = hallu_detect(transcript=transcript_in , audio=audio_in)

    print(result)

if __name__ == "__main__":
    main()