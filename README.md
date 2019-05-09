# HyperGraph Toolkit 

## Overview
**HyperGraph** is an experimental computing framework that implements a polyhedral dynamics approach to the study of ontological assemblages and artifical intelligence. In particular, HyperGraph provides a software implementation of Johnson's hypernetwork theory (Johnson 2014; 2016) and Atkin's Q-analysis (Atkin 1974; 1977). HyperGraph was built using [Python 3.6](https://www.python.org), [Cython 0.20](http://cython.org/), and makes heavy use of [Numpy](http://www.numpy.org/), [Scipy](https://scipy.org/) and [Numba](https://numba.pydata.org/) libaries. 

The toolkit provides methods for collecting data from open media sources such as websites, RSS, and Wikipedia. There are also some tools for accessing Twitter data, Yelp and Reddit. The Spacy.io NLP framework provides options for natural language processing. The toolkit provides additional methods built on Spacy for generating representations for language classification and topic analysis. 

> **Note:** New features include a three multi-criteria decision analysis algorithms that use Q-analysis to rank decision alternatives. Several new notebooks have been added, including the WikiGraph notebook


## Installation

    $ git clone https://github.com/dr-jgsmith/HyperGraph-Project

    $ virtualenv myHyperGraph
MacOS 10.13

     $ source myHyperGraph/bin/activate
Windows

    $ myHyperGraph\Scripts\activate

### CD and Pip

    $ cd /path/to/your/HyperGraph
    $ pip install .


Users will need to install Spacy.io separately, following the steps provided in the spacy.io documentation.


**Run**

* pip install .
* pip install -U spacy
* python -m spacy.en.download --force

**Done!!!**
