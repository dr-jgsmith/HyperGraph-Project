from hypergraphtk.dataminer.wikidata import wikidata
from hypergraphtk.storage.hyperdb import hyperdb
from hypergraphtk.text.txtprocessor import TxtProcessor
from hypergraphtk.core.hyper_graph import *


"""
The script is an example application for creating knowledge representations
from Wikipedia articles and link lists.

The functions collect wikipedia entries, parse the entries, extract phrases
and tokens and then provides a counting of the phrases and tokens.

Phrases and values are then uploaded to the hyperdb. 
"""


wiki = wikidata()
process = TxtProcessor()
db = hyperdb()


def phrase_ranker(content):
    # Phrase ranker ranks phrases in a document
    if content:
        # convert content to lower case
        norm = content.lower()
        # remove new lines and replace with a space.
        data = norm.replace('\n', '')
        # remove special characters
        process.atomize(data)
        clean = process.tokens()
        # load clean data into spacy
        process.load_sequence(clean)
        # extract a count phrases
        process.tokenize()
        ranked_phrases = process.rank_phrases()
        return ranked_phrases
    else:
        return None


def word_ranker(content):
    # ranks words in a document - bag of words representation
    if content:
        # convert content to lower case
        norm = content.lower()
        # remove new lines and replace with a space.
        data = norm.replace('\n', ' ')
        # remove special characters
        process.atomize(data)
        clean = process.tokens()
        # load clean data into spacy
        process.load_sequence(clean)
        # extract and count words
        # x = process.extract_tokens_np()
        x = process.extract_tokens_pos()
        ranked_words = process.rank_words(x)
        return ranked_words
    else:
        return None


def word_phrase_ranker(content):
    # ranks words in a document - bag of words representation
    if content:
        # convert content to lower case
        norm = content.lower()
        # remove new lines and replace with a space.
        data = norm.replace('\n', ' ')
        # remove special characters
        process.atomize(data)
        clean = process.tokens()
        # load clean data into spacy
        process.load_sequence(clean)
        # extract and count words
        ranked_word_phrases = process.rank_word_phrases()
        return ranked_word_phrases
    else:
        return None


def rank_wiki_phrases(topic):
    # Get wikipedia page content for a given topic
    content = wiki.get_page(topic)
    # Encode topic using the word_encoder or the phrase_encoder
    data = phrase_ranker(content)
    if data:
        for i in data.items():
            # data_row = ['wiki_graph', topic, i[0], i[1]]
            db.add_hyperedge('wiki_graph', topic, i[0], i[1])
        return wiki.page.links
    else:
        return None


def rank_wiki_words(topic):
    # Get wikipedia page content for a given topic
    content = wiki.get_page(topic)
    # Encode topic using the word_encoder or the phrase_encoder
    data = word_ranker(content)
    print(data)
    if data:
        for i in data.items():
            # data_row = ['wiki_graph', topic, i[0], i[1]]
            db.add_hyperedge('wiki_graph', topic, i[0], i[1])
        return wiki.page.links
    else:
        return None


def rank_wiki_word_phrases(topic):
    # Get wikipedia page content for a given topic
    content = wiki.get_page(topic)
    # Encode topic using the word_encoder or the phrase_encoder
    data = word_phrase_ranker(content)
    print(data)
    if data:
        for i in data.items():
            # data_row = ['wiki_graph', topic, i[0], i[1]]
            db.add_hyperedge('wiki_graph', topic, i[0], i[1])
        return wiki.page.links
    else:
        return None


def search_list(topic_list):
    topics = []
    for topic in topic_list:
        print(topic)
        data = rank_wiki_words(topic)
        [topics.append(link.lower()) for link in data]

    topic_set = sorted(set(topics))
    for j in topic_set:
        print(j)
        rank_wiki_words(j)

    return


