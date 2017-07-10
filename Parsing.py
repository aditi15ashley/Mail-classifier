from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    

    f.seek(0)  # go back to beginning of file
    all_text = f.read()

    # split off metadata
    content = all_text.split("abcde")
    words = ""
    # remove punctuation
    text_string = content[0].translate(string.maketrans("", ""), string.punctuation)

    # split the text string into individual words, stem each word.
    # space between each stemmed word
    stemmer = SnowballStemmer("english")
    split = text_string.split()
    text = [stemmer.stem(word) for word in split]
    words = ' '.join(text)

    return words


def main():
    ff = open("../Mail_Classifier/Train_Mail.txt", "r")
    text = parseOutText(ff)
    g = open("../Mail_Classifier/Parsedtext.txt", "a")
    g.write(text)
    g.write("\n")
    g.close()

