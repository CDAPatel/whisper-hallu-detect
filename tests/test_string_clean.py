"""
 Placeholder description.
"""

from whisper.hallu_detect import clean_string

def main():
    print("This is a script to test the string cleaner component of the hallucination detector.")

    prompt = "Enter the string to be cleaned: "

    while True:
        text = input(prompt)

        clean, flag = clean_string(text)

        if flag:
                print("Detected an invalid character.")
        else:
            print("Cleaned string:")
            print(clean)



if __name__ == "__main__":
    main()