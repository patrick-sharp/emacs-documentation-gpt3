import pandas as pd
import openai
import numpy as np
import pickle
from transformers import GPT2TokenizerFast

from common import *

corpus_paragraphs = pd.read_csv(CORPUS_PATH)
with open(EMBEDDINGS_PATH, "rb") as input_file:
    corpus_embeddings = pickle.load(input_file)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, corpus_embeddings: list[list[float]]) -> list[(float, str)]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    similarities = []
    for i, row in corpus_paragraphs.iterrows():
        paragraph_text = row[2]
        paragraph_embedding = corpus_embeddings[i]
        similarities.append((vector_similarity(query_embedding, paragraph_embedding), paragraph_text))
    similarities.sort(reverse=True)
    return similarities

sims = order_document_sections_by_query_similarity("What is the shortcut to find a file?", corpus_embeddings)

print(sims[0])

