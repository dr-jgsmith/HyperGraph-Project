import wikipedia
import time

class wikidata:
    
    def __init__(self, ):
        # The WikiData class provides an interface to the wikipedia Python module.
        '''
        Basic usage: 
        data = wikidata()
        content = data.get_page('climate change')
        link_set = data.links
        '''
        # Set some common variables as empty strings.
        self.summary = ''
        self.page = ''
        self.links = ''
        
    #Search wikipedia and get entry summary.
    def get_summary(self, wikiterm):
        split_term = wikiterm.replace('_', ' ')
        try:
            self.summary = wikipedia.summary(split_term)
            time.sleep(10)
            return self.summary
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass
        except wikipedia.exceptions.WikipediaException:
            time.sleep(100)
            pass
        
    def get_page(self, wikiterm):
        split_term = wikiterm.replace('_', ' ')
        try:
            self.page = wikipedia.page(split_term)
            self.links = self.page.links
            time.sleep(10)
            return self.page.content
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass
        except wikipedia.exceptions.WikipediaException:
            time.sleep(100)
            pass

    def get_links(self, wikiterm):
        split_term = wikiterm.replace('_', ' ')
        try:
            self.links = wikipedia.page(split_term)
            time.sleep(10)
            return self.links.links
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass
        except wikipedia.exceptions.WikipediaException:
            time.sleep(100)
            pass
        
    def get_sections(self, wikiterm):
        split_term = wikiterm.replace('_', ' ')
        try:
            self.links = wikipedia.page(split_term)
            time.sleep(10)
            return self.links.sections
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass
        except wikipedia.exceptions.WikipediaException:
            time.sleep(100)
            pass
        
    def get_section_text(self, section):
        try:
            return self.links.section(section)
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass
        except wikipedia.exceptions.WikipediaException:
            time.sleep(100)
            pass
        