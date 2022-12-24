# Script used to extract the text used for the emacs-documentation.txt file.

from bs4 import BeautifulSoup
import csv
import urllib.request

url = urllib.request.urlopen('https://www.gnu.org/software/emacs/manual/html_mono/emacs.html')
content = url.read()
soup = BeautifulSoup(content, 'html.parser')

# write the contents of the <p> tags into a file
with open("emacs-documentation.txt", "w") as out_txt:
    table = soup.findAll('p')
    for p_tag in table:
        out_txt.write(p_tag.text)

# Write each block of text (paragraph, not to be confused with <p> tag)
# into a pandas dataframe.
# A paragraph will contain the text from multiple <p> tags
with (open("emacs-documentation.txt", "r") as in_txt,
      open("emacs-documentation.csv", "w") as out_csv):
    wr = csv.writer(out_csv, quoting=csv.QUOTE_ALL)
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
            wr.writerow([current_paragraph])
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
