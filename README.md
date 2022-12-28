# What this is

This repo is a python program that uses GPT-3 to answer questions about emacs the text editor.

It uses the OpenAI python API.

# How to use

1. Log in to OpenAI here: [https://beta.openai.com/login/](https://beta.openai.com/login/)
2. Acquire an API key here: [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)
3. Put the API key in a file called `openai_api_key` in the root of this repo.
4. Make a python virtual environment with your favorite environment manager (e.g. venv or conda) and install the project's dependencies in that virtual environment.
For example:
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
5. Edit the call to the `answer_question` function at the bottom of the file `answer-question.py` to contain the question you want to ask.
6. Run `python answer-question.py` to see GPT3's answer to your question.

