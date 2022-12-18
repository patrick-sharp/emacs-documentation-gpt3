from bs4 import BeautifulSoup
import urllib.request

url = urllib.request.urlopen('https://www.gnu.org/software/emacs/manual/html_mono/emacs.html')
content = url.read()
soup = BeautifulSoup(content, 'html.parser')

with open("emacs-documentation.txt", "w") as out_txt:
    table = soup.findAll('p')
    for p_tag in table:
        out_txt.write(p_tag.text)

