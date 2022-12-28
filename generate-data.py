# This script is used to extract information from the emacs manual
# and get it into a format ready for use with OpenAI's embeddings
# and GPT-3

# I already used to generate these files:

# emacs-documentation.txt
# emacs-documentation.csv
# embeddings.pickle

# If you have those files (and you should - I included them in the repo),
# then there is no reason to run this file

from bs4 import BeautifulSoup
import csv
import urllib.request
import openai
# The GPT-2 tokenizer is the same as the GPT-3 tokenizer
# https://beta.openai.com/docs/guides/embeddings/frequently-asked-questions
from transformers import GPT2TokenizerFast
import math
import time
import pickle

from common import *

# Get all the text in <p> tags in the emacs manual's html page.
# Generates emacs-documentation.txt
def extract_p_tags():
    url = urllib.request.urlopen('https://www.gnu.org/software/emacs/manual/html_mono/emacs.html')
    content = url.read()
    soup = BeautifulSoup(content, 'html.parser')
    
    # write the contents of the <p> tags into a file
    with open("emacs-documentation.txt", "w") as out_txt:
        table = soup.findAll('p')
        for p_tag in table:
            out_txt.write(p_tag.text)
    
# Now that we have the actual text of the p tags, sort that text into blocks of appropriate size.
# Write each block of text (paragraph, not to be confused with <p> tag)
# into a CSV file.
# A paragraph will contain the text from multiple <p> tags, and be less than MAX_PARAGRAPH_TOKENS
# tokens after being tokenized by the GPT2 tokenizer.
# Generates emacs-documentation.csv
def split_text_into_paragraphs():
    class ParagraphWriter():
        def __init__(self):
            self.rows_written = 0
            self.paragraph_token_nums = []
            self.csv_writer = None

        def write_paragraph(self, paragraph):
            num_tokens = len(tokenizer.tokenize(paragraph))
            self.paragraph_token_nums.append(num_tokens)

            self.csv_writer.writerow([self.rows_written, num_tokens, paragraph])
            self.rows_written += 1;

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    writer = ParagraphWriter()
    with (open("emacs-documentation.txt", "r") as in_txt,
          open(CORPUS_PATH, "w") as out_csv):
        writer.csv_writer = csv.writer(out_csv, quoting=csv.QUOTE_ALL)
        writer.csv_writer.writerow(["Rows written", "Length in tokens", "Paragraph text"])
        lines_in_current_paragraph = []
        lines = in_txt.readlines()
        for line in lines:
            if line == '\n':
                # When we reach the end of the paragraph, append a single string 
                # containing all of the paragraph's text to the paragraphs list
                if len(lines_in_current_paragraph) == 0:
                    # don't add empty rows to the csv
                    continue
                current_paragraph = " ".join(lines_in_current_paragraph)
                num_tokens = len(tokenizer.tokenize(current_paragraph))
                if num_tokens > MAX_PARAGRAPH_TOKENS:
                    # This is a shit show.
                    # If a paragraph is too long, we have to cut it into multiple.
                    # We don't want to split up sentences.
                    # Add sentences to the current paragraph, but not more than MAX_PARAGRAPH_TOKENS total.
                    delimiter = ". "
                    sentences = [e+delimiter for e in current_paragraph.split(delimiter) if e]
                    sentence_lens = []
                    for sentence in sentences:
                        sentence_len = len(tokenizer.tokenize(sentence))
                        if sentence_len > MAX_PARAGRAPH_TOKENS:
                            raise Exception(f"A sentence is longer than {MAX_PARAGRAPH_TOKENS} tokens. That should not happen")
                        sentence_lens.append(sentence_len)
                    token_total = 0
                    start = 0
                    for i, sentence_len in enumerate(sentence_lens):
                        if token_total + sentence_len > MAX_PARAGRAPH_TOKENS:
                            paragraph = "".join(sentences[start:i])
                            writer.write_paragraph(paragraph)
                            token_total = sentence_len
                            start = i
                        elif i == len(sentence_lens):
                            paragraph = "".join(sentences[start:])
                            writer.write_paragraph(paragraph)
                        else:
                            token_total += sentence_len
                else:
                    writer.write_paragraph(current_paragraph)
                lines_in_current_paragraph = []
            else:
                # Check if there is a header for the paragraph. This header contains the
                # name of the previous topic and the next topic, which could confuse the
                # embeddings.search
                split = line.split("[Contents][Index]")
                if len(split) > 1:
                    lines_in_current_paragraph.append(split[1].strip())
                else:
                    lines_in_current_paragraph.append(line.strip())

            #embedding = get_embedding(paragraph)
            #print(type(embedding))
            #self.paragraph_embeddings.append(embedding)

# Now that we have the paragraphs, we need to get the embeddings for them.
# One small problem - OpenAI's free API is quite aggressively rate limited.
# This can fail due to rate limiting. It will save what it has.
# When you run it again, it will pick up where it left off.
# Experimentally, it seems to be able to calculate about 50 each run.
# CSV has 776 rows at time of writing (may change if manual changes), so 
# that should take about 16 runs separated by 30 sec each to get them all.
# Generates embeddings.pickle
def calculate_embeddings():
    # Try to load embeddings from a cached file
    try:
        with open(EMBEDDINGS_PATH, "rb") as input_file:
            paragraph_embeddings = pickle.load(input_file)
    except (TypeError, FileNotFoundError):
        # If there are no embeddings saved, just make an empty list
        paragraph_embeddings = []
    cached_paragraph_embeddings_len = len(paragraph_embeddings)

    with open(CORPUS_PATH, "r") as input_file:
        reader = csv.reader(input_file, quoting=csv.QUOTE_ALL)
        for i, row in enumerate(reader):
            if i < cached_paragraph_embeddings_len + 1: # +1 to skip csv header
                continue
            paragraph = row[2]
            try:
                paragraph_embeddings.append(get_embedding(paragraph))
            except openai.error.RateLimitError:
                print("RateLimitError")
                break

    print("started with", cached_paragraph_embeddings_len, "embeddings.")
    print("exiting with", len(paragraph_embeddings), "embeddings.")
    with open(EMBEDDINGS_PATH, "wb") as file:
        pickle.dump(paragraph_embeddings, file)

# Run the first two functions once, then comment them out and just
# run the last one until you get all the embeddings.
# This shell command might be helpful
# while true; do python generate-data.py; sleep 60; done

# extract_p_tags()
# split_text_into_paragraphs()
calculate_embeddings()

