# SmolLm 360M Chatbot with Voice Support

A chatbot implementation that uses the SmolLm 360M model locally with voice input/output capabilities. This chatbot allows you to have interactive conversations with a lightweight language model that can run on consumer hardware.

The chatbot supports:
- Text-based input and output
- Voice input using OpenAI's Whisper speech recognition
- Voice output using Piper text-to-speech

I've configured this to run on a Raspberry Pi 5 so it can answer questions from elementary age students, but it could be modified to run on anything for anyone.

## Requirements

### Basic Requirements
- Python 3.7+
- PyTorch
- Transformers library
- Hugging Face Hub
- Accelerate library

### Voice Support Requirements
- SoundDevice: For audio recording and playback
- PyAudio: For audio input/output
- NumPy and SciPy: For audio processing
- openai-whisper: For speech recognition (using OpenAI's Whisper model)
- Piper-TTS: For text-to-speech conversion
- SoundFile: For reading/writing audio files
- Microphone and speakers/headphones for voice interaction

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Voice Support Setup

#### PyAudio Installation Note

PyAudio may require additional system dependencies depending on your platform:

- **Linux (Ubuntu/Debian)**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **macOS**: `brew install portaudio`
- **Windows**: No additional dependencies, but you may need Microsoft Visual C++ Build Tools

If you encounter issues installing PyAudio through pip, you can try installing it using conda:
```
conda install pyaudio
```

#### Setting up Whisper (for voice input)

The Whisper model will be downloaded automatically when you first use voice input. Alternatively, you can download it in advance:

```
# Download the Whisper model for offline use
python chatbot.py --download-whisper
```

#### Setting up Piper (for voice output)

To use voice output, you need to download a Piper voice model:

1. Create a directory for Piper voice models:
   ```
   mkdir -p piper-voices
   ```

2. Download a voice model from the [Piper releases page](https://github.com/rhasspy/piper/releases):
   ```
   # Example for downloading the en_US-lessac-medium voice
   wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en_US-lessac-medium.tar.gz
   tar -xzf voice-en_US-lessac-medium.tar.gz -C piper-voices
   ```

3. The voice model includes:
   - `MODEL_NAME.onnx`: The voice model file
   - `MODEL_NAME.onnx.json`: Model configuration
   - `config.json`: Voice configuration

## Usage

### Downloading the Model for Offline Use

To use the chatbot on a device without internet access (like a Raspberry Pi 5), you need to download the model in advance on a machine with internet access:

```
# Download the default model
python chatbot.py --download

# Or specify a different model
python chatbot.py --download --model HuggingFaceTB/SmolLM2-360M

# Specify a custom output directory
python chatbot.py --download --output-dir /path/to/models
```

This will download the model files and save them locally. You can then transfer these files to your offline device.

### Option 1: Using the Python script directly

Run the chatbot with the default settings (requires internet connection for first run):

```
python chatbot.py
```

Run the chatbot with a locally downloaded model (works offline):

```
python chatbot.py --model-dir models/SmolLM2-360M
```

### Option 2: Using the shell script (recommended)

A shell script is provided for convenience:

```
# With internet connection (for first run)
./run_chatbot.sh

# With locally downloaded model (works offline)
./run_chatbot.sh --model-dir models/SmolLM2-360M
```

This script will:
1. Activate the virtual environment if it exists
2. Install requirements if they haven't been installed yet
3. Run the chatbot with any provided arguments

The chatbot will:
1. Load the model (from local directory if specified, otherwise download from Hugging Face)
2. Start an interactive chat session in the terminal

Type your messages and press Enter to get a response from the chatbot. Type 'exit', 'quit', or 'bye' to end the conversation.

## Command-line Arguments

The chatbot supports the following command-line arguments:

### Language Model Arguments
- `--model`: Specify a different model ID to use (default: "HuggingFaceTB/SmolLM-360M-Instruct")
- `--device`: Specify the device to run the model on ('cpu', 'cuda', 'mps')
- `--model-dir`: Directory containing the local model files (for offline use)
- `--download-lm`: Download the language model for offline use (replaces the previous `--download` flag)
- `--output-dir`: Directory to save the downloaded models (default: "models")

### Voice Input Arguments
- `--voice-input`: Enable voice input using Whisper
- `--whisper-model`: Specify a different Whisper model size (default: "small", options: "tiny", "base", "small", "medium", "large")
- `--whisper-model-dir`: Directory containing the local Whisper model files (for offline use)
- `--download-whisper`: Download the Whisper model for offline use

### Voice Output Arguments
- `--voice-output`: Enable voice output using Piper
- `--piper-voice`: Specify the Piper voice to use (default: "en_US-lessac-medium")
- `--piper-model-dir`: Directory containing the Piper voice models (default: "piper-voices")

Examples:

Using the Python script directly:

```
# Basic usage (text only)
python chatbot.py

# Run on CPU explicitly
python chatbot.py --device cpu

# Use a different model
python chatbot.py --model HuggingFaceTB/SmolLM2-360M

# Download language model for offline use
python chatbot.py --download-lm --model HuggingFaceTB/SmolLM2-360M

# Run with a locally downloaded model (offline mode)
python chatbot.py --model-dir models/SmolLM2-360M

# Enable voice input
python chatbot.py --voice-input

# Enable voice output
python chatbot.py --voice-output

# Enable both voice input and output
python chatbot.py --voice-input --voice-output

# Download Whisper model for offline voice input
python chatbot.py --download-whisper

# Use a specific Piper voice
python chatbot.py --voice-output --piper-voice en_GB-alba-medium

# Full offline mode with voice support
python chatbot.py --model-dir models/SmolLM2-360M --voice-input --voice-output --whisper-model-dir models/whisper-small
```

Using the shell script:

```
# Basic usage (text only)
./run_chatbot.sh

# Run on CPU explicitly
./run_chatbot.sh --device cpu

# Use a different model
./run_chatbot.sh --model HuggingFaceTB/SmolLM2-360M

# Download language model for offline use
./run_chatbot.sh --download-lm --model HuggingFaceTB/SmolLM2-360M

# Run with a locally downloaded model (offline mode)
./run_chatbot.sh --model-dir models/SmolLM2-360M

# Enable voice input
./run_chatbot.sh --voice-input

# Enable voice output
./run_chatbot.sh --voice-output

# Enable both voice input and output
./run_chatbot.sh --voice-input --voice-output

# Full offline mode with voice support
./run_chatbot.sh --model-dir models/SmolLM2-360M --voice-input --voice-output --whisper-model-dir models/whisper-small
```

## Model Information

The default model used is SmolLm 360M, a lightweight language model with 360 million parameters. This model is small enough to run on most consumer hardware while still providing reasonable conversational capabilities.

## Notes

### General Notes
- The first run will download the model if `--model-dir` is not specified, which may take some time depending on your internet connection.
- For offline use (e.g., on a Raspberry Pi 5 without internet), download the model in advance using the `--download-lm` flag and then use `--model-dir` to specify the local model directory.
- The language model requires approximately 1GB of RAM/VRAM to run.
- For better performance, running on a GPU is recommended but not required.
- On a Raspberry Pi 5, the model will run in CPU mode and may be slower than on more powerful hardware, but it should still be functional.

### Voice Feature Notes
- Voice input requires a working microphone connected to your device.
- Voice output requires speakers or headphones connected to your device.
- The Whisper model requires additional RAM/VRAM to run (approximately 500MB for the small model).
- Piper voice models are typically 40-100MB each, depending on the voice quality.
- On a Raspberry Pi 5, voice processing may be slower, but it should still be usable.
- For offline use with voice features, download both the language model and the Whisper model in advance.
- If you encounter audio input/output issues, check your system's audio settings and ensure the microphone and speakers are properly configured.
- The default recording duration for voice input is 5 seconds. You can modify this in the code if needed.
