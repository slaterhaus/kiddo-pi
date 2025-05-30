#!/usr/bin/env python3
"""
SmolLm 360M Chatbot

A simple chatbot implementation using the SmolLm 360M model locally.
"""

import os
import argparse
from typing import Optional, List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList

# Model identifier for SmolLm 360M
MODEL_ID = "HuggingFaceTB/SmolLM-360M-Instruct"  # 360M parameter model from the SmolLM family
DEFAULT_MODEL_DIR = "models"  # Default directory to store downloaded models

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


def truncate_response(response, max_sentences=3):
    sentences = response.split('. ')
    return '. '.join(sentences[:max_sentences]) + '.' if sentences else response


class Chatbot:
    def __init__(self, model_id=MODEL_ID, device=None, model_dir=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            max_new_tokens=256,
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
        print("SmolLM Chatbot - Type 'exit' to quit")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                messages.append({"role": "user", "content": f"{user_input}\n\nAnswer in 2–3 sentences for a child"})

                response = self.generate_response(messages)
                print(f"\nAssistant: {response}")

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
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID to use")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--model-dir", type=str, help="Directory containing the local model files")
    parser.add_argument("--download", action="store_true", help="Download the model for offline use")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory to save the downloaded model")
    args = parser.parse_args()

    # Download the model if requested
    if args.download:
        model_dir = download_model(args.model, args.output_dir)
        print(f"Model downloaded to {model_dir}. You can use it offline with --model-dir={model_dir}")
        return

    # Initialize and start chatbot
    chatbot = Chatbot(model_id=args.model, device=args.device, model_dir=args.model_dir)
    chatbot.chat()


if __name__ == "__main__":
    main()
