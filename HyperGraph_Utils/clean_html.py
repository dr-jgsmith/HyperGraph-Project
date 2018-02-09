#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""************************************************************************************
Created on Tue Jul 25 09:50:32 2017

@author: justinsmith


    Baseline html tag removal. Should be called all documents to ensure special characters 
    are removed. Works relatively well for our currrent purposes. It is an older method 
    originally supplied within the NLTK package but was removed in the most recent iterations. 
    I saved it because it actually works very well compared to other methods I have seen.
    
************************************************************************************"""

import re

def clean_html(html):
    """
    Remove HTML markup from the given string.
    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """
    str_html = str(html)
    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", str_html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"[\s]", "  ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", "\n", cleaned)

    clean = cleaned.split()

    for i in clean:
        if len(i) <= 1:
            clean.remove(i)
        else:
            pass
    clean = ' '.join(clean)
    return clean


    
        