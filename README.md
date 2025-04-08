# Whisper Audio Transcription App

A Windows GUI application for transcribing audio files using OpenAI's Whisper speech recognition model locally on your machine.

## Features

- Transcribe audio files (MP3, WAV, M4A, FLAC, OGG, MP4)
- Select from different Whisper model sizes (tiny, base, small, medium, large)
- Choose between CPU or GPU processing (if CUDA is available)
- Progress indicator during transcription
- View raw transcript results
- Convert to Markdown with live preview
- Export as raw text or Markdown files
- Custom save location

## Installation

1. Clone or download this repository
2. Create and activate a virtual environment:
```
python -m venv whisper_env
whisper_env\Scripts\activate
```
3. Install required dependencies:
```
pip install -r requirements.txt
```

Note: The Whisper package is installed directly from the GitHub repository to ensure compatibility.

## Creating the Executable

1. Make sure you have PyInstaller installed:
```
pip install pyinstaller
```

2. Create the executable using the provided spec file:
```
pyinstaller whisper_transcribe.spec
```

3. The executable will be created in the `dist` folder. You can create a shortcut to `dist/WhisperTranscribe/WhisperTranscribe.exe` on your desktop.

Note: The first time you run the executable, it will download the Whisper model files. This might take a few minutes depending on your internet connection and the model size you select.

## Usage

1. Run the application (either through Python or the executable)
2. Select an audio file to transcribe
3. Choose the Whisper model size and processing device
4. Set your preferred save directory (optional)
5. Click "Transcribe Audio" and wait for the process to complete
6. View the results in the Raw Transcript tab
7. Optionally convert to Markdown with the "Prettify to Markdown" button
8. Export the transcription as a text or Markdown file

## Requirements

- Python 3.7+
- PyQt6
- OpenAI Whisper (from GitHub)
- PyTorch
- Markdown

## Notes

- Larger models provide better transcription quality but require more memory and processing time
- GPU acceleration significantly improves processing speed for larger models
- The application creates "transcribed_text" and "uploaded_audio" directories in the application folder
- The executable includes all necessary dependencies and will work on any Windows system 