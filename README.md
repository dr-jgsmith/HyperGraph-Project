# HyperGraph: An Embedded Distributed Application Framework for the Study and Design of Intelligent Systems

## Overview
Improvements in computational processing speed has unleashed a flood of research into  the application of topology to a range of complex, n-dimensional systems problems. This has lead to major breakthroughs in medical diagnostics, computer graphics, decision-science, systems design, and artificial intelligence. Despite a growing body of research stretching back to the 1930s, there are surprisingly few computing frameworks that specifically support topological data analysis (TDA). 

**HyperGraph** is a lightweight, experimental computing framework that implements a polyhedral dynamics approach to the study of ontological assemblages and artifical intelligence. In particular, HyperGraph provides a software implementation of Johnson's hypernetwork theory (Johnson 2014; 2016) and Atkin's Q-analysis (Atkin 1974; 1977). HyperGraph’s use of simplicial families and complexes, makes it suitable as a general computing framework for studying a diverse set of complex systems, including natural language. 

HyperGraph has already been used for automated ontology generation, text classification and sentiment analysis, and it is being extended as a 'smart' complex event detection system to monitor disaster events (and risk) around the world.

**Keywords:** *Hypernetworks, Q-Analysis, Topology, Complexity Science, Natural Language Processing, Artificial Intelligence*

> **Note:** This repo only contains the core functionality of the HyperGraph System. This includes the core algorithims used to compute and process hypegraphs, hypernetworks and perform Q-analysis on those data types.
> 	

## Current Version v.001
HyperGraph was built using [Python 3.6](https://www.python.org), [Cython 0.20](http://cython.org/), and tested on Mac and Windows based operating systems. The main components of HyperGraph include the Hypernet Algorithm, Encoder, and an embedded key-value datastore ([Vedis v.0.90,](http://vedis.symisc.net/) [LevelDB v.1.18,](http://leveldb.org/) or [RocksDB](http://rocksdb.org/)), that can be configured to use in-memory, disk and/or remote data storage as part of a networked solution. HyperGraph also comes with a simple TCP/IP client-server, as well as a number of data mining resources that can be used to collect and process Wikipedia entries, Twitter, Facebook, Yelp, and RSS feeds (currently includes feeds for financial markets, and emergency alerts around the world), as well as finance and industry data such as agricultural commodities, cryptocurrency markets, and more (additional data can be included).

Because HyperGraph's first use was for natural language processing (NLP), it is designed to play nice with existing Python based NLP systems (e.g. [Spacy.io](https://spacy.io) [Gensim,](https://radimrehurek.com/gensim/) and [NLTK](https://nltk.org)), in particular HyperGraph relies on the Spacy.io [maximum entropy](http://www.aclweb.org/anthology/W16-2703) (MaxEnt) model for tagging parts of speech, and Gensim’s implementation of the latent Dirichlet allocation (LDA) algorithm. The HyperGraph leverages these  packages to construct topological representations that use both structural and semantic features in a data stream. 

The hypernet algorithm is another feature that comes with the HyperGraph system. Hypernet was developed spefically for handling topological data. The algorithm can be used for data reduction, ranked preference modeling, single-link and agglomerative clustering, sequence co-expression and weighted-correlation networks, sentiment analysis, and classification. A number of machine learning packages are also included, and more can be installed to extend the capabilities of a HyperGraph node (e.g. [Scikit-Learn](http://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org), [Keras](https://keras.io/)). Currently, Scikit-learn,and the Scipy package are the only libraries that come preinstalled, but HyperGraph has been tested with [Keras](https://keras.io/) for building long short-term memory (LSTM) based neural networks.


## HyperGraph Core
