#!/bin/bash

# Change to the project root directory where run_sup_cls.py is located
cd "$(dirname "$0")/.."

# Run classification on new data
echo "--- Poem ---"
echo "ChatGPT"
python run_sup_cls.py --human data/poem_nll/poem_Human_llama3-8b-instruct.txt --model data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.txt
echo "GPT3"
python run_sup_cls.py --human data/poem_nll/poem_Human_llama3-8b-instruct.txt --model data/poem_nll/poem_GPT3_poem_llama3-8b-instruct.txt
echo "Llama2-70B-chat"
python run_sup_cls.py --human data/poem_nll/poem_Human_llama3-8b-instruct.txt --model data/poem_nll/poem_Llama2-70B-chat_poem_llama3-8b-instruct.txt
echo "Olmo-7B-instruct"
python run_sup_cls.py --human data/poem_nll/poem_Human_llama3-8b-instruct.txt --model data/poem_nll/poem_Olmo-7B-instruct_poem_llama3-8b-instruct.txt
echo "Tulu2-dpo-70B"
python run_sup_cls.py --human data/poem_nll/poem_Human_llama3-8b-instruct.txt --model data/poem_nll/poem_Tulu2-dpo-70B_poem_llama3-8b-instruct.txt
echo "------------"

echo "--- Book ---"
echo "ChatGPT"
python run_sup_cls.py --human data/book_nll/Human_book_llama3-8b-instruct.txt --model data/book_nll/ChatGPT_book_llama3-8b-instruct.txt
echo "GPT3"
python run_sup_cls.py --human data/book_nll/Human_book_llama3-8b-instruct.txt --model data/book_nll/GPT3_book_llama3-8b-instruct.txt
echo "Llama2-70B-chat"
python run_sup_cls.py --human data/book_nll/Human_book_llama3-8b-instruct.txt --model data/book_nll/Llama2-70B-chat_book_llama3-8b-instruct.txt
echo "Olmo-7B-instruct"
python run_sup_cls.py --human data/book_nll/Human_book_llama3-8b-instruct.txt --model data/book_nll/Olmo-7B-instruct_book_llama3-8b-instruct.txt
echo "Tulu2-dpo-70B"
python run_sup_cls.py --human data/book_nll/Human_book_llama3-8b-instruct.txt --model data/book_nll/Tulu2-dpo-70B_book_llama3-8b-instruct.txt
echo "------------"

echo "--- Speech ---"
echo "ChatGPT"
python run_sup_cls.py --human data/speech_nll/Human_speech_llama3-8b-instruct.txt --model data/speech_nll/ChatGPT_speech_llama3-8b-instruct.txt
echo "GPT3"
python run_sup_cls.py --human data/speech_nll/Human_speech_llama3-8b-instruct.txt --model data/speech_nll/GPT3_speech_llama3-8b-instruct.txt
echo "Llama2-70B-chat"
python run_sup_cls.py --human data/speech_nll/Human_speech_llama3-8b-instruct.txt --model data/speech_nll/Llama2-70B-chat_speech_llama3-8b-instruct.txt
echo "Olmo-7B-instruct"
python run_sup_cls.py --human data/speech_nll/Human_speech_llama3-8b-instruct.txt --model data/speech_nll/Olmo-7B-instruct_speech_llama3-8b-instruct.txt
echo "Tulu2-dpo-70B"
python run_sup_cls.py --human data/speech_nll/Human_speech_llama3-8b-instruct.txt --model data/speech_nll/Tulu2-dpo-70B_speech_llama3-8b-instruct.txt
echo "------------"

echo "Done"