#!/usr/bin/env python3
"""
SmolLm 360M Chatbot

A simple chatbot implementation using the SmolLm 360M model locally.
"""

import os
# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import tempfile
import ssl
import urllib.request
from typing import List, Dict

import torch
import sounddevice as sd
import soundfile as sf
import whisper
import re
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList


def disable_ssl_verification():
    """
    Disable SSL certificate verification for urllib.request.
    This is needed to handle self-signed certificates when downloading models.
    """
    # Create a custom SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Patch urllib.request to use our custom SSL context
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)

# Model identifier for SmolLm 360M
MODEL_ID = "HuggingFaceTB/SmolLM-360M-Instruct"  # 360M parameter model from the SmolLM family
DEFAULT_MODEL_DIR = "models"  # Default directory to store downloaded models
DEFAULT_WHISPER_MODEL = "small"  # Default Whisper model size
DEFAULT_RECORDING_DURATION = 5  # Default recording duration in seconds
SAMPLE_RATE = 16000  # Sample rate for audio recording

messages = [
    {
        "role": "system",
        "content": "You are a friendly elementary school teacher. Use simple, clear language for children ages 5–10. Explain things step by step, use fun examples, and encourage curiosity. Always be positive and supportive. If you don’t know something, say so and suggest where to find the answer. Make answer short unless otherwise asked for long answers. Make learning fun and interactive!"
    }
]


def download_model(model_id: str, output_dir: str) -> str:
    """
    Download a model from Hugging Face and save it locally.

    Args:
        model_id: The Hugging Face model ID to download
        output_dir: The directory to save the model to

    Returns:
        The path to the downloaded model
    """
    # Create the output directory if it doesn't exist
    model_dir = os.path.join(output_dir, os.path.basename(model_id))
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading model {model_id} to {model_dir}...")

    # Download the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Save the model and tokenizer to the output directory
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    print(f"Model downloaded and saved to {model_dir}")
    return model_dir


class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences):
        self.stop_token_ids = [tokenizer.encode(stop, add_special_tokens=False) for stop in stop_sequences]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False


def record_audio(duration=DEFAULT_RECORDING_DURATION):
    """
    Record audio from the microphone for the specified duration.

    Args:
        duration: Recording duration in seconds

    Returns:
        numpy.ndarray: Recorded audio data
    """
    print(f"\nListening for {duration} seconds...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio_data


def transcribe_audio(audio_data, whisper_model=None):
    """
    Transcribe audio data to text using Whisper.

    Args:
        audio_data: Audio data as numpy array
        whisper_model: Whisper model to use for transcription

    Returns:
        str: Transcribed text
    """
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        sf.write(temp_file.name, audio_data, SAMPLE_RATE)

        # Load Whisper model if not provided
        if whisper_model is None:
            # Disable SSL verification to handle self-signed certificates
            disable_ssl_verification()

            # Load the Whisper model with SSL verification disabled
            whisper_model = whisper.load_model(DEFAULT_WHISPER_MODEL)

        # Transcribe audio
        result = whisper_model.transcribe(temp_file.name, fp16=False)

    return result["text"].strip()


def download_whisper_model(model_name=DEFAULT_WHISPER_MODEL, output_dir=DEFAULT_MODEL_DIR):
    """
    Download a Whisper model for offline use.

    Args:
        model_name: Name of the Whisper model to download
        output_dir: Directory to save the model to

    Returns:
        str: Path to the downloaded model
    """
    whisper_dir = os.path.join(output_dir, f"whisper-{model_name}")
    os.makedirs(whisper_dir, exist_ok=True)

    print(f"Downloading Whisper {model_name} model...")

    # Disable SSL verification to handle self-signed certificates
    disable_ssl_verification()

    # Load the Whisper model with SSL verification disabled
    model = whisper.load_model(model_name)

    print(f"Whisper model downloaded. It will be cached at {os.path.expanduser('~/.cache/whisper')}")
    return whisper_dir


def truncate_response(response, max_sentences=3):
    sentences = response.split('. ')
    return '. '.join(sentences[:max_sentences]) + '.' if sentences else response


def speak_text(text):
    """
    Convert text to speech using pyttsx3.

    Args:
        text: The text to convert to speech
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


class Chatbot:
    def __init__(self, model_id=MODEL_ID, device=None, model_dir=None,
                 voice_input=False, whisper_model_name=DEFAULT_WHISPER_MODEL, whisper_model_dir=None,
                 voice_output=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.voice_input = voice_input
        self.voice_output = voice_output
        self.whisper_model = None

        # Load Whisper model if voice input is enabled
        if self.voice_input:
            print(f"Loading Whisper {whisper_model_name} model for speech recognition...")

            # Disable SSL verification to handle self-signed certificates
            disable_ssl_verification()

            # Load the Whisper model with SSL verification disabled
            self.whisper_model = whisper.load_model(whisper_model_name)
            print("Whisper model loaded.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir or model_id,
            revision="v0.1",  # Use improved v0.2 version
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir or model_id,
            revision="v0.1",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )

        # Set up stopping criteria
        stop_sequences = ["\nUser:", "\nuser:", "User:", "user:"]
        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.tokenizer, stop_sequences)
        ])

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            stopping_criteria=stopping_criteria,
            # device=self.device,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.9,
            top_k=40,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def format_prompt(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate_response(self, conversation_history: List[Dict]) -> str:
        formatted_prompt = self.format_prompt(conversation_history)

        response = self.generator(
            formatted_prompt,
            return_full_text=False,
            clean_up_tokenization_spaces=True
        )[0]['generated_text']

        # Remove any trailing stop sequences
        for stop_seq in ["\nUser:", "\nuser:"]:
            if stop_seq in response:
                response = response.split(stop_seq)[0].strip()

        return truncate_response(response.strip())

    def chat(self):
        global messages
        voice_status = ""
        if self.voice_input:
            voice_status += " (Voice input enabled)"
        if self.voice_output:
            voice_status += " (Voice output enabled)"

        print("SmolLM Chatbot - Type 'exit' to quit" + voice_status)
        while True:
            try:
                if self.voice_input:
                    # Record audio and transcribe it
                    audio_data = record_audio()
                    user_input = transcribe_audio(audio_data, self.whisper_model, )
                    print(f"\nYou (voice): {user_input}")

                    # If user didn't say anything, continue listening
                    if not user_input.strip():
                        print("No voice input detected. Listening again...")
                        continue
                else:
                    # Use text input
                    user_input = input("\nYou: ")

                if re.search('^(exit|bye|quit).?$', user_input, re.IGNORECASE):
                    break

                messages.append({"role": "user", "content": f"{user_input}\n\nAnswer in 2–3 sentences for a child, no more than 128 tokens."})

                response = self.generate_response(messages)
                print(f"\nAssistant: {response}")

                # Use text-to-speech if enabled
                if self.voice_output:
                    speak_text(response)

                messages.append({"role": "assistant", "content": response})

                if len(messages) > 9:
                    messages = [messages[0]] + messages[-8:]

            except KeyboardInterrupt:
                break


def main():
    """
    Main function to parse arguments and start the chatbot.
    """
    parser = argparse.ArgumentParser(description="SmolLm 360M Chatbot")

    # Language model arguments
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID to use")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--model-dir", type=str, help="Directory containing the local model files")
    parser.add_argument("--download-lm", action="store_true", help="Download the language model for offline use")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory to save the downloaded models")

    # Voice input arguments
    parser.add_argument("--voice-input", action="store_true", help="Enable voice input using Whisper")
    parser.add_argument("--voice-output", action="store_true", help="Enable voice output using text-to-speech")
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--whisper-model-dir", type=str, help="Directory containing the local Whisper model files")
    parser.add_argument("--download-whisper", action="store_true", help="Download the Whisper model for offline use")

    # For backward compatibility
    parser.add_argument("--download", action="store_true",
                        help="Download the language model for offline use (deprecated, use --download-lm)")

    args = parser.parse_args()

    # Handle backward compatibility
    if args.download:
        args.download_lm = True
        print("Warning: --download is deprecated, use --download-lm instead")

    # Download the language model if requested
    if args.download_lm:
        model_dir = download_model(args.model, args.output_dir)
        print(f"Model downloaded to {model_dir}. You can use it offline with --model-dir={model_dir}")
        return

    # Download the Whisper model if requested
    if args.download_whisper:
        whisper_dir = download_whisper_model(args.whisper_model, args.output_dir)
        print(f"Whisper model downloaded. You can use it with --voice-input --whisper-model={args.whisper_model}")
        return

    # Initialize and start chatbot
    chatbot = Chatbot(
        model_id=args.model,
        device=args.device,
        model_dir=args.model_dir,
        voice_input=args.voice_input,
        whisper_model_name=args.whisper_model,
        whisper_model_dir=args.whisper_model_dir,
        voice_output=args.voice_output
    )
    chatbot.chat()


if __name__ == "__main__":
    main()
