import feedparser


class rsstreams:
    # URL = None
    # URL_AUTHOR = None
    # Needs work

    def __init__(self, list_sources=None, add_sources_name=None, add_source_url=None):
        """
            Empty intializer.
            This will change with the addition of class options.
            e.g options will include the ability to add feedlinks and feed titles.
        """
        self.list_sources = list_sources
        self.add_sources = add_sources_name
        self.add_source_url = add_source_url

        # List of RSS feeds that we will fetch and combine
        self.newsurls = {
            'apnews': 'http://hosted2.ap.org/atom/APDEFAULT/3d281c11a76b4ad082fe88aa0db04909',
            'googlenews': 'https://www.google.com/alerts/feeds/04804902391332952922/16891434162289931789',
            'reutersBiz': 'http://feeds.reuters.com/reuters/businessNews',
            'energynews': 'http://fuelfix.com/blog/author/bloomberg/feed/',
            'earnings': 'http://www.cnbc.com/id/15839135/device/rss/rss.html',
            'finance': 'http://www.cnbc.com/id/10000664/device/rss/rss.html',
            'asiamarkets': 'http://www.intellasia.net/category/financeasia/feed',
            'asiancommodities': 'http://www.intellasia.net/category/resourceasia/feed',
            'yahoonews': 'http://news.yahoo.com/rss/',
            'global_disasters': 'http://www.gdacs.org/xml/rss.xml',
            'reutersMoney': 'http://feeds.reuters.com/news/wealth',
            'reutersEnv': 'http://feeds.reuters.com/reuters/environment',
            'reutersTech': 'http://feeds.reuters.com/reuters/technologyNews',
            'reutersSci': 'http://feeds.reuters.com/reuters/scienceNews',
            'NWS_Alerts': 'https://alerts.weather.gov/cap/us.php?x=0'
        }

    # This function calls the FeedParser Lib to execute the rss parsing.
    def parseRSS(self, rss_url):
        
        
        return feedparser.parse(rss_url)

        # Function grabs the rss feed headlines (titles) and returns them as a list

    def getHeadlines(self, rss_url):

        self.headlines = {}

        self.feed = self.parseRSS(rss_url)
        for newsitem in self.feed['items']:
            self.headlines['data_type'] = 'rss'
            self.headlines['tag'] = 'tag'
            self.headlines['title'] = newsitem['title']
            self.headlines['link'] = newsitem['link']
            self.headlines['text'] = newsitem['summary']
            self.headlines['published'] = newsitem['published']

        # print('Printing subset of headlines...', set(headlines))
        return self.headlines

    # A list to hold all headlines
    def getRssFeed(self):
        allheadlines = []
        # Iterate over the feed urls
        for key, url in self.newsurls.items():
            # Call getHeadlines() and combine the returned headlines with allheadlines
            #allheadlines.extend(self.getHeadlines(url))
            x = self.getHeadlines(url)
            x['tag'] = key
            allheadlines.append(x)
            
        self.news_list = allheadlines
    
        return self.news_list

    

    
    


