import openai

openai.api_key_path = "./openai_api_key"


EMBEDDINGS_MODEL = "text-embedding-ada-002" # This is the one OpenAI recommends for almost all use cases as of 2022-12-05
COMPLETIONS_MODEL = "text-davinci-003"


MAX_PROMPT_TOKENS = 4000 # This is the maximum length of a GPT-3 prompt
MAX_PARAGRAPH_TOKENS = 3800 # This leaves 200 tokens left over for your question

CORPUS_PATH = "emacs-documentation.csv"
EMBEDDINGS_PATH = "embeddings.pickle"

def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(
      model=EMBEDDINGS_MODEL,
      input=text
    )
    return result["data"][0]["embedding"]
