#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:20:59 2017

@author: justinsmith
"""

from __future__ import print_function


import json
import csv
from twitter import *
import time
from geopy.geocoders import Nominatim

geolocator = Nominatim()

twit_config = {
            "access_token": 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
            "access_secert": 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
            "consumer_key": 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
            "consumer_secret": 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        }

def search_twitter(twit_config, search_list):
    oauth = OAuth(twit_config["access_token"], twit_config["access_secert"], twit_config["consumer_key"], twit_config["consumer_secret"])
    t = Twitter(auth=oauth)
    
    for item in search_list:
        iterate = t.search.tweets(q=item, count=100, lang='en')
        for result in iterate["statuses"]:
            row = {"data_type": "twitter", "tag": item, "author": result["user"]["screen_name"], "text": result["text"]}

            if result["geo"] and result["entities"]["urls"]:
                link = result["entities"]["urls"][0]
                row['link'] =  link.get("expanded_url")
                row["location"] = result["user"]["location"]
                loc = {"latitude": result["geo"]["coordinates"][0],'longitude': result["geo"]["coordinates"][1]}
                row["geolocation"] = loc
            elif result["entities"]["urls"]:
                link = result["entities"]["urls"][0]
                row['link'] =  link.get("expanded_url")
                row["location"] = result["user"]["location"]
                row["geolocation"] = ''
            elif result:
                row['link'] =  "none"
                row["location"] = result["user"]["location"]
                row["geolocation"] = ''
              
            row['date'] = result["created_at"]
            print(row)
            
            
def search_twitter_deep(twit_config, search_list):
    oauth = OAuth(twit_config["access_token"], twit_config["access_secert"], twit_config["consumer_key"], twit_config["consumer_secret"])
    t = Twitter(auth=oauth)
    out = open('festival_data.csv', 'a')
    outfile = csv.writer(out)
    
    for item in search_list:
        num_results = 100
        result_count = 0
        last_id = None
        while result_count <  num_results:
            time.sleep(5)
            iterate = t.search.tweets(q=item, count=100, lang='en', max_id=last_id)
            for result in iterate["statuses"]:
                row = {"data_type": "twitter", "tag": item, "author": result["user"]["screen_name"], "text": result["text"]}
    
                if result["geo"]:
                        tag = item 
                        author = result["user"]["screen_name"]
                        text = result["text"]
                        text = text.encode('ascii', 'replace')
                        if result["entities"]["urls"]:
                            link = result["entities"]["urls"][0]
                            url =  link.get("expanded_url")
                        else:
                            url = ' '
                        loc = result["user"]["location"]
                        latitude = result["geo"]["coordinates"][0]
                        longitude = result["geo"]["coordinates"][1]
                        date = result["created_at"]
                        row = [tag, author, text, url, loc, latitude, longitude, date]
                elif result["user"]["location"]:
                        tag = item 
                        author = result["user"]["screen_name"]
                        text = result["text"]
                        text = text.encode('ascii', 'replace')
                        
                        if result["entities"]["urls"]:
                            link = result["entities"]["urls"][0]
                            url =  link.get("expanded_url")
                        else:
                            url = ' '
                            
                        loc = result["user"]["location"]
                        latitude = ' '
                        longitude = ' '
                        date = result["created_at"]
                        row = [tag, author, text, url, loc, latitude, longitude, date]
                elif result:
                        tag = item 
                        author = result["user"]["screen_name"]
                        text = result["text"]
                        if result["entities"]["urls"]:
                            link = result["entities"]["urls"][0]
                            url =  link.get("expanded_url")
                        else:
                            url = ' '
                        loc = ' '
                        
                        date = result["created_at"]
                        row = [tag, author, text, url, loc, latitude, longitude, date]
                else:
                        pass
                
                outfile.writerow(row)
                result_count += 1
                if last_id == result["id"]:
                    result_count = num_results+1
                else:
                    last_id = result["id"]
                print(row, result_count)
                
#New set of methods for doing some targeted data mining on twitter by location
def search_twitter_loc(twit_config, search_list, radius=15):
    oauth = OAuth(twit_config["access_token"], twit_config["access_secert"], twit_config["consumer_key"], twit_config["consumer_secret"])
    t = Twitter(auth=oauth)
    
    file = open('data/usplaces_fix2.csv', 'rt')
    places = csv.reader(file)
    
    out = open('data/festival_data10.csv', 'a')
    outfile = csv.writer(out)
    
    next(places)
    for place in places:
        if place[12] == ' ':
            pass
        else:
            if int(place[12]) < 30000:
                pass
            else:
                lat = place[0]
                lon = place[1]
                city = place[9]
                state = place[13]
                print(city, state)
                for item in search_list:
                    time.sleep(5)
                    query = t.search.tweets(q=item, geocode="%f,%f,%dmi" % (float(lon), float(lat), radius), count=100, lang='en')
                    for result in query["statuses"]:
                        if result["geo"]:
                            tag = item 
                            author = result["user"]["screen_name"]
                            text = result["text"]
                            text = text.encode('ascii', 'replace')
                            
                            if result["entities"]["urls"]:
                                link = result["entities"]["urls"][0]
                                url =  link.get("expanded_url")
                            else:
                                url = ' '
                            
                            loc = result["user"]["location"]
                            latitude = result["geo"]["coordinates"][0]
                            longitude = result["geo"]["coordinates"][1]
                            date = result["created_at"]
                            row = [tag, author, text, url, loc, latitude, longitude, date]
                        
                        elif result["user"]["location"]:
                            tag = item 
                            author = result["user"]["screen_name"]
                            text = result["text"]
                            text = text.encode('ascii', 'replace')
                            
                            if result["entities"]["urls"]:
                                link = result["entities"]["urls"][0]
                                url =  link.get("expanded_url")
                            else:
                                url = ' '
                                
                            loc = result["user"]["location"]
                            latitude = lat
                            longitude = lon
                            date = result["created_at"]
                            row = [tag, author, text, url, loc, latitude, longitude, date]
                        
                        elif result:
                            tag = item 
                            author = result["user"]["screen_name"]
                            text = result["text"]
                            if result["entities"]["urls"]:
                                link = result["entities"]["urls"][0]
                                url =  link.get("expanded_url")
                            else:
                                url = ' '
                            loc = city + " " + state
                            latitude = lat
                            longitude = lon
                            date = result["created_at"]
                            row = [tag, author, text, url, loc, latitude, longitude, date]
                        else:
                            pass
                    
                        print(row)
                        outfile.writerow(row)
                
        #return self.result 
        
search = ['cherry blossom', 'festival']#['tulip festival', 'festival']
search_twitter_loc(twit_config, search)