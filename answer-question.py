import pandas as pd
import openai
import numpy as np
import pickle
from transformers import GPT2TokenizerFast

from common import *

class Corpus():
    def __init__(self):
        self.paragraphs = pd.read_csv(CORPUS_PATH)
        with open(EMBEDDINGS_PATH, "rb") as f:
            self.embeddings = pickle.load(f)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(
    query: str, 
    corpus: Corpus
) -> list[(float, int, int, str)]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    similarities = []
    for i, row in corpus.paragraphs.iterrows():
        paragraph_token_len = row[1]
        paragraph_text = row[2]
        paragraph_embedding = corpus.embeddings[i]
        similarities.append(
            (
                vector_similarity(query_embedding, paragraph_embedding),
                i,
                paragraph_token_len,
                paragraph_text
            )
        )
    similarities.sort(reverse=True)
    return similarities

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

def construct_prompt(
    question: str, 
    corpus: Corpus
) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, 
        corpus)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for i, (_, par_idx, par_len, par_text)  in enumerate(most_relevant_document_sections):
        # Add contexts until we run out of space.        
        if chosen_sections_len + par_len > MAX_PARAGRAPH_TOKENS:
            break
        chosen_sections_len += par_len + separator_len
            
        chosen_sections.append(SEPARATOR + par_text)
        chosen_sections_indexes.append(str(par_idx))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections with {chosen_sections_len} tokens total:")
    print("\n".join(chosen_sections_indexes), "\n")
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_question(
    query: str,
    corpus: Corpus,
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(query, corpus)
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    answer = response["choices"][0]["text"].strip(" \n")
    print(answer)
    return answer


corpus = Corpus()

# answer_question("What is the shortcut to open a split window to the right?", corpus)
answer_question("What is the shortcut to view my open buffers?", corpus)
