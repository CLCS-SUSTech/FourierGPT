#!/bin/bash

# FFT processing script for book_nll, poem_nll, and speech_nll datasets
# This script processes all .txt files in the specified directories using run_fft.py

# Change to the project root directory where run_fft.py is located
cd "$(dirname "$0")/.."

echo "Starting FFT processing for all datasets..."

# Process book_nll dataset
echo "Processing book_nll dataset..."
python run_fft.py -i data/book_nll/ChatGPT_book_llama3-8b-instruct.txt -o data/book_nll/ChatGPT_book_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/book_nll/GPT3_book_llama3-8b-instruct.txt -o data/book_nll/GPT3_book_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/book_nll/Human_book_llama3-8b-instruct.txt -o data/book_nll/Human_book_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/book_nll/Llama2-70B-chat_book_llama3-8b-instruct.txt -o data/book_nll/Llama2-70B-chat_book_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/book_nll/Olmo-7B-instruct_book_llama3-8b-instruct.txt -o data/book_nll/Olmo-7B-instruct_book_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/book_nll/Tulu2-dpo-70B_book_llama3-8b-instruct.txt -o data/book_nll/Tulu2-dpo-70B_book_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore

# Process poem_nll dataset
echo "Processing poem_nll dataset..."
python run_fft.py -i data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.txt -o data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/poem_nll/poem_GPT3_poem_llama3-8b-instruct.txt -o data/poem_nll/poem_GPT3_poem_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/poem_nll/poem_Human_llama3-8b-instruct.txt -o data/poem_nll/poem_Human_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/poem_nll/poem_Llama2-70B-chat_poem_llama3-8b-instruct.txt -o data/poem_nll/poem_Llama2-70B-chat_poem_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/poem_nll/poem_Olmo-7B-instruct_poem_llama3-8b-instruct.txt -o data/poem_nll/poem_Olmo-7B-instruct_poem_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/poem_nll/poem_Tulu2-dpo-70B_poem_llama3-8b-instruct.txt -o data/poem_nll/poem_Tulu2-dpo-70B_poem_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore

# Process speech_nll dataset
echo "Processing speech_nll dataset..."
python run_fft.py -i data/speech_nll/ChatGPT_speech_llama3-8b-instruct.txt -o data/speech_nll/ChatGPT_speech_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/speech_nll/GPT3_speech_llama3-8b-instruct.txt -o data/speech_nll/GPT3_speech_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/speech_nll/Human_speech_llama3-8b-instruct.txt -o data/speech_nll/Human_speech_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/speech_nll/Llama2-70B-chat_speech_llama3-8b-instruct.txt -o data/speech_nll/Llama2-70B-chat_speech_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/speech_nll/Olmo-7B-instruct_speech_llama3-8b-instruct.txt -o data/speech_nll/Olmo-7B-instruct_speech_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore
python run_fft.py -i data/speech_nll/Tulu2-dpo-70B_speech_llama3-8b-instruct.txt -o data/speech_nll/Tulu2-dpo-70B_speech_llama3-8b-instruct.nllzs.fftnorm.txt -p zscore

echo "FFT processing completed for all datasets!"
