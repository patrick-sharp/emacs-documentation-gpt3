import openai

openai.api_key_path = "./openai_api_key"


EMBEDDINGS_MODEL = "text-embedding-ada-002" # This is the one OpenAI recommends for almost all use cases as of 2022-12-05
COMPLETIONS_MODEL = "text-davinci-003"


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

SEPARATOR = "\n* "

MAX_CONTEXT_TOKENS = 4097 # This is the maximum length of a GPT-3 prompt + completions
MAX_PROMPT_TOKENS = MAX_CONTEXT_TOKENS - COMPLETIONS_API_PARAMS["max_tokens"]
MAX_PARAGRAPH_TOKENS = MAX_PROMPT_TOKENS - 400 # Leaves room for the separators and question

CORPUS_PATH = "emacs-documentation.csv"
EMBEDDINGS_PATH = "embeddings.pickle"

def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(
      model=EMBEDDINGS_MODEL,
      input=text
    )
    return result["data"][0]["embedding"]
