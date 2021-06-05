# Write your code here

from nltk.tokenize import word_tokenize
from collections import Counter
from lxml import etree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import nltk

xml_path = "news.xml"
tree = etree.parse(xml_path)
root = tree.getroot()


def tokenize_text(text, datasets):

    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tagged_tokens = [nltk.pos_tag([token])[0] for token in lemmatized_tokens]

    noun_tokens = [tagged_token[0] for tagged_token in tagged_tokens
                     if tagged_token[1] == 'NN']

    excludes = stopwords.words('english') + list(string.punctuation)
    filtered_tokens = list(
        filter(lambda x: x not in excludes, noun_tokens))

    datasets.append(" ".join(filtered_tokens))


def parse_xml(element, datasets, headlines):
    for child in element:

        if child.get("name") == "head":
            headlines.append(child.text)
        if child.get("name") == "text":
            tokenize_text(child.text, datasets)

        parse_xml(child, datasets, headlines)


def print_top5(datasets, headlines):
    vectorizer = TfidfVectorizer(input='content',
                                 use_idf=True,
                                 analyzer='word',
                                 ngram_range=(1, 1)
                                 )

    tfidf_matrix = vectorizer.fit_transform(datasets)
    tfidf_matrix = tfidf_matrix.toarray()

    for row, headline in enumerate(headlines):
        print(headline + ":")

        frequencies = dict()

        for column, token in enumerate(vectorizer.get_feature_names()):
            # print(token, row, column, tfidf_matrix[row][column])
            if tfidf_matrix[row][column]:
                frequencies[token] = tfidf_matrix[row][column]

        sorted_frequencies = dict(
            sorted(frequencies.items(),
                   key=lambda item: (item[1], item[0]),
                   reverse=True
                   ))

        top5_tokens = [x for x in sorted_frequencies.keys()][0:5]
        print(*top5_tokens, sep=" ")


all_headlines = []
all_datasets = []

parse_xml(root, all_datasets, all_headlines)
print_top5(all_datasets, all_headlines)
