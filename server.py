import json
import pdb

import torch
from flask import Flask, jsonify, request
# from spacy.lang.en import English
# from spacy.tokenizer import Tokenizer
from torch import nn
from transformers import (BertConfig, BertModel, BertTokenizer,
                          BertTokenizerFast, DistilBertConfig, DistilBertModel,
                          DistilBertTokenizerFast, FeatureExtractionPipeline)

adaptive_pool = nn.AdaptiveAvgPool1d(100)

# nlp = English()
# tokenizer = nlp.Defaults.create_tokenizer(nlp)

bc = BertConfig(output_hidden_states=True)
# db = BertModel.from_pretrained("bert-base-multilingual-cased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert = bert.eval()
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_tok = BertTokenizer("./bert-base-uncased-vocab.txt")
# dbt = BertTokenizerFast("./bert-base-multilingual-cased-vocab.txt")
# senteceembedder = FeatureExtractionPipeline(db, dbt)

app = Flask(__name__)


@app.route('/tokenize',  methods=['POST'])
def tokenize():
    utterances = request.json["utterances"]

    tokenized = []
    for sentence in utterances:
        # ### Approche par spacy
        # tokens = [t.text for t in tokenizer(sentence)]

        # ### Approche avec distilBert
        tokens = []
        for t in bert_tok.tokenize(sentence):
            if t.startswith("##"):
                if tokens:
                    tokens[-1] += t[2:]
            else:
                tokens.append(t)

        tokenized.append(tokens)

    return jsonify(tokenized)


@app.route('/vectorize',  methods=['POST'])
def vectorize():
    tokens = request.json["tokens"]
    # print("\n\n\n", tokens, "\n\n\n")
    embeddings = []
    with torch.no_grad():
        for tok in tokens:
            encoded = bert_tok.encode(tok, return_tensors="pt")
            out_tensor, hidden = bert(encoded)

            # out_tensor = adaptive_pool(out_tensor)
            # hidden = adaptive_pool(hidden.unsqueeze(0))

            # out_tensor = torch.sum(out_tensor, axis=1).squeeze()
            hidden = hidden.squeeze()
            assert len(hidden.size()) == 1
            embeddings.append(hidden.detach().cpu().data.numpy().tolist())

    # input_tensor = bert_tok.batch_encode_plus(tokens, pad_to_max_length=True,
    #                                           return_tensors="pt")
    # outputs = bert(input_tensor["input_ids"],
    #                input_tensor["attention_mask"])
    # outputs = adaptive_pool(outputs[0])
    # outputs = outputs[0]
    # embeddings = outputs.squeeze().cpu().data.numpy().tolist()
    assert len(embeddings) == len(tokens)
    for e in embeddings:
        assert len(e) == 768
    return jsonify({"vectors": embeddings})


@app.route('/vectorize_utterances',  methods=['POST'])
def vectorize_utterances():
    utterances = request.json["utterances"]
    with torch.no_grad():
        input_tensor = bert_tok.batch_encode_plus(utterances,
                                                  pad_to_max_length=True,
                                                  return_tensors="pt")

        outputs, hidden = bert(input_tensor["input_ids"],
                               input_tensor["attention_mask"])

        # outputs = adaptive_pool(outputs[0])
        # embedding = torch.sum(outputs, axis=1)
        # embedding = adaptive_pool(hidden.unsqueeze(0)).squeeze()
        embedding = hidden
        embeddings = embedding.detach().cpu().data.numpy().tolist()
    return jsonify({"vectors": embeddings})


@app.route('/info',  methods=['GET'])
def info():
    infos = {
        "version": "1",
        "ready": True,
        "dimentions": 768,
        "domain": "bp",
        "readOnly": True,
        "languages": [
            {"lang": "en", "loaded": True},
        ]
    }
    return jsonify(infos)


if (__name__ == '__main__'):
    app.run(debug=True)


# Botpress tokenizer mon_vectorize

# Botpress tokenizer mon_sentence

# size 100 / 300 / 786


# Test 1 vectorize 768 bert puis botpress
# Test 2 vectorize 300 bert puis botpress
# Test 3 vectorize 100 bert puis botpress

# Test 4 tokenize botpress puis vecotrize sentence bert 768
# Test 4 tokenize botpress puis vecotrize sentence bert 300
# Test 4 tokenize botpress puis vecotrize sentence bert 100
