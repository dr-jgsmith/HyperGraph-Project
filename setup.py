from setuptools import setup

setup(name='hypergraphtk',
      version='0.02',
      description='Toolkit for computing with hypergraphs and hypernetworks',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Text Processing :: Linguistic :: Complex Systems :: Topology',
      ],
      url='https://github.com/dr-jgsmith/HyperGraph-Project',
      author='Justin G. Smith',
      author_email='justingriffis@wsu.edu',
      license='MIT',
      packages=['hypergraphtk',
                'hypergraphtk.core',
                'hypergraphtk.dataminer',
                'hypergraphtk.storage',
                'hypergraphtk.scripts',
                'hypergraphtk.text',
                'hypergraphtk.visual'],
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
      include_package_data=True,
      zip_safe=False)
