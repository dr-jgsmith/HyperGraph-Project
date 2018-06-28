import requests
import lxml
from bs4 import BeautifulSoup


class crawler:

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
                if data is None:
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
