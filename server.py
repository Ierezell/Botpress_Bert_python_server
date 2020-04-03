import json

import torch
from flask import Flask, jsonify, request
from spacy.lang.fr import French
from spacy.tokenizer import Tokenizer
from transformers import (DistilBertConfig, DistilBertModel,
                          DistilBertTokenizerFast, FeatureExtractionPipeline)
from torch import nn
adaptive_pool = nn.AdaptiveAvgPool1d(300)
nlp = French()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

dc = DistilBertConfig(output_hidden_states=True)
db = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
dbt = DistilBertTokenizerFast("./bert-base-multilingual-cased-vocab.txt")
senteceembedder = FeatureExtractionPipeline(db, dbt)

app = Flask(__name__)


@app.route('/tokenize',  methods=['POST'])
def tokenize():
    print("\n\n\n\n\n", request.json, "\n\n\n\n")
    import pdb
    pdb.set_trace()
    return None

    utterances = request.json["utterances"]

    tokenized = []
    for sentence in utterances:
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

        tokenized.append(tokens)

    return jsonify(tokenized)


@app.route('/vectorize',  methods=['POST'])
def vectorize():
    # print("\n\n\n\n\n", request.json, "\n\n\n\n")
    # import pdb
    # pdb.set_trace()
    tokens = request.json["tokens"]
    input_tensor = dbt.batch_encode_plus(tokens, pad_to_max_length=True,
                                         return_tensors="pt")

    outputs = db(input_tensor["input_ids"],
                 input_tensor["attention_mask"])

    outputs = adaptive_pool(outputs[0])
    embeddings = outputs.squeeze().cpu().data.numpy().tolist()
    return jsonify(embeddings)


@app.route('/vectorize_utterances',  methods=['POST'])
def vectorize_utterances():
    utterances = request.json["utterances"]
    # print("\n\n\n\n", utterances, "\n\n\n")
    # import pdb
    # pdb.set_trace()
    input_tensor = dbt.batch_encode_plus(utterances, pad_to_max_length=True,
                                         return_tensors="pt")

    outputs = db(input_tensor["input_ids"],
                 input_tensor["attention_mask"])

    outputs = adaptive_pool(outputs[0])
    embedding = torch.sum(outputs, axis=1)
    embeddings = embedding.cpu().data.numpy().tolist()
    return jsonify({"vectors": embeddings})


@app.route('/info',  methods=['GET'])
def info():
    infos = {
        "version": "1",
        "ready": True,
        "dimentions": 300,
        "domain": "bp",
        "readOnly": True,
        "languages": [
            {"lang": "en", "loaded": True},
        ]
    }
    return jsonify(infos)

if (__name__ == '__main__'):
    app.run(debug=True)