#!/bin/bash

# Script to run the SmolLm 360M Chatbot with Voice Support
# Supports both online and offline modes, with text and voice interaction
# 
# For offline use (e.g., on a Raspberry Pi 5 without internet):
#   1. Download the models on a machine with internet:
#      ./run_chatbot.sh --download-lm
#      ./run_chatbot.sh --download-whisper (if using voice input)
#   2. Download Piper voice models (if using voice output):
#      See README.md for instructions
#   3. Transfer the model files to the offline device
#   4. Run with the local models:
#      ./run_chatbot.sh --model-dir models/SmolLM2-360M
#      
# For voice support:
#   - Enable voice input: ./run_chatbot.sh --voice-input
#   - Enable voice output: ./run_chatbot.sh --voice-output
#   - Enable both: ./run_chatbot.sh --voice-input --voice-output

# Check if virtual environment exists and activate it if it does
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if requirements are installed
if [ ! -f ".requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch .requirements_installed
fi

# Run the chatbot with all provided arguments
echo "Starting SmolLm 360M Chatbot..."
python chatbot.py "$@"

# Deactivate virtual environment if it was activated
if [ -d ".venv" ]; then
    deactivate
fi
