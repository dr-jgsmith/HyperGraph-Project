import spacy


class TxtProcessor:
    def __init__(self):
        # Initialize a Spacy object
        self.nlp = spacy.load('en')

    # The methods below are a list of helper functions to split up a stream of text.
    # Atomize and tokens are optional standalone tools. They can be used
    # with or without the Spacy.io framework.
    def atomize(self, text):
        # Get a stream of text
        self.text = text
        # break the data into their lowest level - single character
        self.txt_atoms = [c for c in text]
        # convert that string to data char to a unicode representation
        self.uni_atoms = [ord(c) for c in self.txt_atoms]
        return

    def tokens(self):
        # define characters that need removed
        # method is uncessary if using the extract_tokens_pos method
        stop_char = ['=', '(', ')', '/', ':', ';', ',', '?', '|' '"', '[', ']']
        clean = [i for i in self.txt_atoms if i not in stop_char]
        cleanwords = ''.join(clean)
        return cleanwords

    # Primary Spacy.io based methods for basic text processing
    # Steps include load text or sequence
    def load_sequence(self, text):
        self.data = self.nlp(text)
        return self.data

    # Tokenization and Stop Word Removal
    def tokenize(self):
        self.tokens = [word.text for word in self.data if word.is_stop is False]
        return self.tokens

    # Parts-of-Speech Tagging
    def pos_tags(self):
        self.tags = [(word.text, word.pos_) for word in self.data]
        return self.tags

    # Lemmatize a sequence of words - Get root word
    def word_roots(self):
        self.roots = [word.lemma_ for word in self.data if word.is_stop is False]
        return self.roots

    # Extract named entities from a text
    def entities(self):
        self.entity = [(word.text, word.label_) for word in self.data.ents]
        return self.entity

    # Extract Subject, Object, Predicate
    def parse_triples(self):
        self.trigram = []
        for np in self.data.noun_chunks:
            struct = (np.root.text, np.root.dep_, np.root.head.text)
            self.trigram.append(struct)
        return self.trigram

    # Simple noun-phrase extraction
    def noun_phrase(self):
        phrase = []
        for np in self.data.noun_chunks:
            struct = (np.text)
            phrase.append(struct)
        return phrase

    # More complex noun-phrase extraction method
    def noun_phrase_re(self):
        clean_phrases = []
        phrase = []
        for np in self.data.noun_chunks:
            struct = (np.text)
            phrase.append(struct)
        for i in phrase:
            data = i.split(' ')
            for j in data:
                if j.is_stop is True:
                    data.remove(j)
                else:
                    pass
            cdata = ' '.join(data)
            clean_phrases.append(cdata)
        return clean_phrases

    # This method extracts words or tokens from the noun_phrase spacy function
    # The results are typically cleaner than the raw tokenizer
    def extract_tokens_np(self):
        clean_tokens = []
        phrases = []
        # iterate through the phrases
        for np in self.data.noun_chunks:
            struct = (np.text)
            phrases.append(struct)
        # Now phrases are string objects instead of spacy objects
        for i in phrases:
            # Iterate and split phrases with more than one word
            j = i.split(' ')
            # Iterate through split phrases
            for k in j:
                # check if word is a stop word, pass if true
                if k.is_stop is True:
                    pass
                else:
                    clean_tokens.append(k)
        return clean_tokens


    # Token extraction using the Spacy POS tagger
    # Perhaps the cleanest method for extracting noun-phrases in text
    # Allows greater control of the types of patterns that can be retained.
    def extract_tokens_pos(self):
        tokens = []
        data = self.pos_tags()
        # Define tags to retain
        tags = ['NOUN', 'VERB', 'ADJ']
        # Define additional stop words on the fly
        stop_fly = ['is', 'be', 'am', 'were', 'been', 'are', 'was', "'s", "'d", "'re", "'t"]
        for i in data:
            if i[1] in tags:
                if i[0] in stop_fly:
                    pass
                else:
                    tokens.append(i[0])
            else:
                pass

        return tokens

    # Rank words using the POS phrase extraction method
    def rank_words_phrases(self):
        tokens = {}
        words = self.extract_tokens_pos()
        phrases = self.noun_phrase()
        word_set = sorted(set(words))
        for i in word_set:
            tokens[i] = words.count(i)
        # This method computes the value of tokens in a phrase
        # Given the new token value the
        for i in phrases:
            j = i.split(' ')
            vals = []
            for k in j:
                val = words.count(k)
                vals.append(val)
            if sum(vals) > 0:
                tokens[i] = str(sum(vals) / len(vals))
            else:
                pass
        return tokens


    # Rank words by occurrence in a text
    def rank_words(self, tokens):
        root_set = sorted(set(tokens))
        counts = {}
        for i in root_set:
            val = tokens.count(i)
            counts[i] = val
        return counts


    # Rank phrases based on occurrence of phrase words throughout a text.
    def rank_phrases(self):
        # Count roots words in the phrase
        lemma_counts = {}
        # Use simple phrase extractor
        phrases = self.noun_phrase()
        # Get the entire set of words and their lemmas
        roots = list(self.word_roots())
        # Count lemmas in the text
        lemma_set = sorted(set(roots))
        for lem in lemma_set:
            val = roots.count(lem)
            lemma_counts[lem] = val
        # Collect phrases
        ranked_phrases = {}
        for j in phrases:
            # load phrase
            self.load_sequence(j)
            # root words in the phrase
            proots = self.word_roots()
            # Check to see if the phrase is an empty string
            if len(proots) > 0:
                # count phrase lemmas in the text
                value = [lemma_counts.get(k) for k in proots]
                if None in value:
                    pass
                else:
                    # connect lemmas to reconstruct phrase
                    phrase = ' '.join(proots)
                    # sum occurrences of lemmas in text and apply value to phrase
                    ranked_phrases[phrase] = str(sum(value) / len(proots))
            else:
                pass

        return ranked_phrases

    # Simply gets the phrases, lemmas and parts-of-speech tags for a sequence of text.
    def get_all_layers(self):
        phrases = self.noun_phrase_re()
        roots = self.word_roots()
        tags = self.pos_tags()
        return phrases, roots, tags
