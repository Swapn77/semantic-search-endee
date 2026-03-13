# Semantic Search Engine using Endee

I built this as part of the Endee internship assignment. The idea is to search documents 
by meaning rather than exact keywords. So if you search "how to stay fit", it returns 
results about exercise, diet, sleep etc. even if none of them contain those exact words.

## What problem does this solve

Normal search engines match keywords exactly. If a document says "cardiovascular exercise" 
and you search "working out", it won't find it. Semantic search fixes this by converting 
text into vectors and comparing meaning instead of words. This is exactly what Endee is 
built for.

## How it works

The user types a query. The sentence-transformers library converts it into a vector. 
The same is done for all documents at startup. Then cosine similarity is used to find 
which documents are closest in meaning. Endee is the vector database responsible for 
storing and searching these vectors efficiently.

## Project structure
```
semantic-search-endee/
├── search.py             terminal version of the search engine
├── app.py                web version using Flask
├── README.md
└── templates/
    └── index.html        frontend UI
```

## How to run

install dependencies first:
```
pip install sentence-transformers numpy flask
```

for the terminal version:
```
python search.py
```

for the web version:
```
python app.py
```

then open your browser and go to http://127.0.0.1:5000

## Some queries to try

- how to stay healthy
- what is machine learning
- famous places in Europe
- climate and environment

## Tech used

- Python
- Endee (vector database)
- sentence-transformers (for generating embeddings)
- NumPy (for cosine similarity)
- Flask (for the web interface)

## Forked from

https://github.com/endee-io/endee