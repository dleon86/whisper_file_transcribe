import whisper
import numpy as np

def transcribe_audio(input_file, output_file='output.txt'):
    try:
        model = whisper.load_model('medium') #medium
        result = model.transcribe(input_file)
        lines = result['text']
        with open(output_file, 'w') as f:
            f.write(lines)
    except FileNotFoundError:
        print(f"Error: The input file {input_file} could not be found.")
    except KeyError:
        print("Error: The audio file provided is not compatible with the transcribe model.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


transcribe_audio(r'C:\Users\danny\Documents\Projects\gpt3_Playground\audio\Isabelle Cohen-Gidon.mp3',
                 r'C:\Users\danny\Documents\Projects\gpt3_Playground\text\Isabelle Cohen-Gidon.txt') 