import json

import torch
from flask import Flask, jsonify, request
from spacy.lang.fr import French
from spacy.tokenizer import Tokenizer
from transformers import (DistilBertConfig, DistilBertModel,
                          DistilBertTokenizerFast, FeatureExtractionPipeline)
from torch import nn
adaptive_pool = nn.AdaptiveAvgPool1d(100)
nlp = French()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

dc = DistilBertConfig(output_hidden_states=True)
db = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
dbt = DistilBertTokenizerFast("./bert-base-multilingual-cased-vocab.txt")
senteceembedder = FeatureExtractionPipeline(db, dbt)

app = Flask(__name__)


@app.route('/tokenize',  methods=['POST'])
def tokenize():
    print(request.text())

    sentence = request.json["sentence"]

    # ### Approche par spacy
    # tokens = [t.text for t in tokenizer(sentence)]

    # ### Approche avec distilBert
    tokens = []
    for t in dbt.tokenize(sentence):
        if t.startswith("##"):
            if tokens:
                tokens[-1] += t[2:]
        else:
            tokens.append(t)

    return jsonify(tokens)


@app.route('/vectorize',  methods=['POST'])
def vectorize():
    sentence = request.json["sentence"]
    input_ids = torch.tensor(dbt.encode(sentence)).unsqueeze(0)
    outputs = db(input_ids)
    outputs = adaptive_pool(outputs[0])
    embeddings = outputs[0].squeeze().cpu().data.numpy().tolist()
    return jsonify(embeddings)


@app.route('/embedsentence',  methods=['POST'])
def embedsentence():
    sentence = request.json["sentence"]
    input_ids = torch.tensor(dbt.encode(sentence)).unsqueeze(0)
    outputs = db(input_ids)
    outputs = adaptive_pool(outputs[0])
    embedding = torch.sum(outputs, axis=1)
    return jsonify(embedding.cpu().data.numpy().tolist())


@app.route('/info',  methods=['GET'])
def info():
    infos = {
        "version": "1",
        "ready": True,
        "dimentions": 100,
        "domain": "bp",
        "readOnly": True,
        "languages": [
            {"lang": "ar", "loaded": True},
            {"lang": "de", "loaded": True},
            {"lang": "en", "loaded": True},
            {"lang": "es", "loaded": True},
            {"lang": "fr", "loaded": True},
            {"lang": "he", "loaded": True},
            {"lang": "it", "loaded": True},
            {"lang": "ja", "loaded": True},
            {"lang": "nl", "loaded": True},
            {"lang": "pl", "loaded": True},
            {"lang": "pt", "loaded": True},
            {"lang": "ru", "loaded": True}
        ]
    }
    return jsonify(infos)
