from hypergraphtk.storage.hyperdb import hyperdb
from hypergraphtk.text.txtprocessor import TxtProcessor
from hypergraphtk.dataminer.wikidata import wikidata


"""
The script is an example application for creating knowledge representations
from Wikipedia articles and link lists.

The functions collect wikipedia entries, parse the entries, extract phrases
and tokens and then provides a counting of the phrases and tokens.

Phrases and values are then uploaded to the hyperdb. 
"""


def load():
    wiki = wikidata()
    process = TxtProcessor()
    hdb = hyperdb()
    return wiki, process, hdb


def phrase_ranker(object, content):
    # Phrase ranker ranks phrases in a document
    if content:
        # convert content to lower case
        norm = content.lower()
        # remove new lines and replace with a space.
        data = norm.replace('\n', '')
        # remove special characters
        object.atomize(data)
        clean = object.tokens()
        # load clean data into spacy
        object.load_sequence(clean)
        # extract a count phrases
        object.tokenize()
        ranked_phrases = object.rank_phrases()
        return ranked_phrases
    else:
        return None


def word_ranker(object, content):
    # ranks words in a document - bag of words representation
    if content:
        # convert content to lower case
        norm = content.lower()
        # remove new lines and replace with a space.
        data = norm.replace('\n', ' ')
        # remove special characters
        object.atomize(data)
        clean = object.tokens()
        # load clean data into spacy
        object.load_sequence(clean)
        # extract and count words
        # x = process.extract_tokens_np()
        x = object.extract_tokens_pos()
        ranked_words = object.rank_words(x)
        return ranked_words
    else:
        return None


def word_phrase_ranker(object, content):
    # ranks words in a document - bag of words representation
    if content:
        # convert content to lower case
        norm = content.lower()
        # remove new lines and replace with a space.
        data = norm.replace('\n', ' ')
        # remove special characters
        object.atomize(data)
        clean = object.tokens()
        # load clean data into spacy
        object.load_sequence(clean)
        # extract and count words
        ranked_word_phrases = object.rank_word_phrases()
        return ranked_word_phrases
    else:
        return None


def rank_wiki_phrases(object, topic):
    # Get wikipedia page content for a given topic
    content = object[0].get_page(topic)
    # Encode topic using the word_encoder or the phrase_encoder
    data = phrase_ranker(object[1], content)
    if data:
        for i in data.items():
            # data_row = ['wiki_graph', topic, i[0], i[1]]
            object[2].add_hyperedge('wiki_graph', topic, i[0], i[1])
        return object[0].page.links
    else:
        return None


def rank_wiki_words(object, topic):
    # Get wikipedia page content for a given topic
    content = object[0].get_page(topic)
    # Encode topic using the word_encoder or the phrase_encoder
    data = word_ranker(object[1], content)
    print(data)
    if data:
        for i in data.items():
            # data_row = ['wiki_graph', topic, i[0], i[1]]
            object[2].add_hyperedge('wiki_graph', topic, i[0], i[1])
        return object[0].page.links
    else:
        return None


def rank_wiki_word_phrases(object, topic):
    # Get wikipedia page content for a given topic
    content = object[0].get_page(topic)
    # Encode topic using the word_encoder or the phrase_encoder
    data = word_phrase_ranker(object[1], content)
    # print(data)
    if data:
        for i in data.items():
            # data_row = ['wiki_graph', topic, i[0], i[1]]
            object[2].add_hyperedge('wiki_graph', topic, i[0], i[1])
        return object[0].page.links
    else:
        return None


def search_list(topic_list):
    dobject = load()
    topics = []
    for topic in topic_list:
        print(topic)
        data = rank_wiki_words(dobject, topic)
        [topics.append(link.lower()) for link in data]

    topic_set = sorted(set(topics))
    for j in topic_set:
        print(j)
        rank_wiki_words(dobject, j)

    return
