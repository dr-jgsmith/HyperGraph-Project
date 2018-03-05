from setuptools import setup

setup(name='hypergraphtk',
      version='0.1',
      description='Toolkit for computing with hypergraphs and hypernetworks',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing :: Linguistic :: Complex Systems :: Topology',
      ],
      url='https://github.com/dr-jgsmith/HyperGraph-Project',
      author='Justin G. Smith',
      author_email='justingriffis@wsu.edu',
      license='MIT',
      packages=['hypergraphtk', 
                'hypergraphtk.hyper_core',
                'hypergraphtk.data_miner',
                'hypergraphtk.storage',
                'hypergraphtk.txtprocessor',
                'hypergraphtk.visualize'],
      install_requires=[
          'spacy',
          'dataset',
          'stuf'
          'seaborn',
          'wikipedia',
          'networkx',
          'lxml',
          'feedparser',
          'twitter'
      ],
      zip_safe=False)