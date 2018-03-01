import requests
import lxml
from bs4 import BeautifulSoup
import scholarly
import wikipedia
import time


class HtmlMapper:

    """*****************************************************************************
        
        HtmlMapper is a simple class and associated methods for crawling and scraping 
        html content. It makes heavy use of Requests and BeautifulSoup.
        Basic usage includes extracting all links from a site.
        Second-order link extraction can be performed with a link extraction of the 
        terminating links.
        
        Usage example:
        import csv
        import time
        from random import randrange
        
        f = open('extension-text.csv', 'w', newline='')
        file = csv.writer(f)
        
        x = HtmlMapper(start_url="http://articles.extension.org/")
        links = x.get_seed_links() #scrapes all of the links from the page
        link_data = x._deep_link_scrape(links)
        
        d1 = []
        for i in link_data:
            
            d1.append(i[0])
            d1.append(i[1])
            dset = sorted(set(d1))
            
        clean_links = []
        for i in dset:
            a = list(i)
            b = len(a) - 2
            c = a[b:]
            d = ''.join(c)
            if d == "//":
                b = len(a) - 1
                c = a[:b]
                print(''.join(c))
                clean_links.append(''.join(c))
            else:
                print(i)
                clean_links.append(i)
        
        e_links = []
        for i in clean_links:
            if 'extension.org' in i:
                e_links.append(i)
            else:
                pass
                
        d2 = []
        for i in sorted(set(e_links)):
            link = i
            print(link)
            text = x.get_text_from_link(link)
            row = [link, text]
            d2.append(row)
            file.writerow(row)
            t = randrange(5)
            time.sleep(t)
                    
        
    *******************************************************************************"""

    def __init__(self, term=None, start_url=None):
        self.start_url = start_url
        self.term = term
        self.edges = []

    def get_seed_links(self):
        self.start_list = []
        self.start_list.append(self.start_url)
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        r = s.get(self.start_url)
        self.c = r.content
        soup = BeautifulSoup(self.c, "lxml")
        for link in soup.find_all("a"):
            data = link.get("href")
            if data == None:
                pass
            elif data[:5] == "https":
                row = (self.start_url, data + "/")
                self.edges.append(row)
                self.start_list.append(data)
            elif data[:4] == "http":
                row = (self.start_url, data + "/")
                self.edges.append(row)
                self.start_list.append(data)
            else:
                pass
        return self.start_list


    # Second order web scrap for link retreival. This function can only work if a link list has been generatated from the previous function call.
    def _deep_link_scrape(self, link_list):
        self.link_list = link_list
        for i in self.link_list:
            s = requests.Session()
            s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
            r = s.get(i, verify=False)
            c = r.content
            soup = BeautifulSoup(c, "lxml")
            for link in soup.find_all("a"):
                data = link.get("href")
                if data == None:
                    pass
                elif data[:5] == "https":
                    row = (i, data + "/")
                    self.edges.append(row)
                elif data[:4] == "http":
                    row = (self.start_url, data + "/")
                    self.edges.append(row)
                else:
                    pass  
        return self.edges

                
    # Simple function for calling a python requests response to render html as a raw/binary output.
    def get_raw_html(self):
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        self.r = s.get(self.start_url)
        return self.r.raw
    
    
    # Simple function for calling a python requests response to render html as a text string.
    def get_data_from_link(self, url):
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        r = s.get(self.start_url)
        self.texts = r.text
        return self.texts
    
     # Simple function for calling a python requests response to render html as a text string.
    def get_text_from_link(self, url):
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        try:
            r = s.get(url, verify=False)
            texts = r.content
            return texts
        except requests.SSLError:
            pass
    
    #Search Google Scholar
    def search_scholar(self):
        search_query = scholarly.search_pubs_query(self.term)
        self.result = next(search_query).fill()
        return self.result


    #Search wikipedia and get entry summary.
    def get_summary(self, wikiterm):
        try:
            split_term = wikiterm.replace('_', ' ')
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
        try:
            split_term = wikiterm.replace('_', ' ')
            self.page = wikipedia.page(split_term)
            time.sleep(5)
            return self.page.content
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass
        except wikipedia.exceptions.WikipediaException:
            time.sleep(100)
            pass
        except AttributeError:
            self.page = wikipedia.page(wikiterm)
            time.sleep(5)
            return self.page.content

    def get_links(self, wikiterm):
        try:
            split_term = wikiterm.replace('_', ' ')
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
        try:
            split_term = wikiterm.replace('_', ' ')
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
        


