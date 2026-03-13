# Semantic Search Engine using Endee

I built this project as part of the Endee internship assignment. The idea is simple — instead of searching for exact words, this engine understands the *meaning* of your query and returns the most relevant results.

For example if you search "how to stay fit", it will return results about exercise, diet, sleep etc. even if none of them contain the words "stay fit".

## What problem does this solve

Normal search engines match keywords. So if a document says "cardiovascular exercise" and you search "working out", it won't find it. Semantic search fixes this by converting text into vectors and comparing meaning instead of words.

## How it works

The user types a query. The sentence-transformers library converts it into a vector (basically a list of numbers that represent meaning). The same was done for all documents when the program starts. Then we use cosine similarity to find which documents are closest in meaning to the query. Endee is the vector database that handles storing and searching these vectors efficiently.

## How to run

make sure you have Python installed, then:
```
pip install sentence-transformers numpy
```

then just run:
```
python search.py
```

## Some queries to try

- how to stay healthy
- what is machine learning  
- famous places in Europe

## Tech used

- Python
- Endee (vector database)
- sentence-transformers (for generating embeddings)
- NumPy (for cosine similarity)

## Forked from

https://github.com/endee-io/endee