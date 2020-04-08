import json
import pdb

import torch
from flair.data import Sentence
from flask import Flask, jsonify, request
from torch import nn
from flair.embeddings import (
    BertEmbeddings,
    # BytePairEmbeddings,
    # CharacterEmbeddings,
    # ELMoEmbeddings,
    # FastTextEmbeddings,
    # FlairEmbeddings,
    # OneHotEmbeddings,
    # OpenAIGPTEmbeddings,
    # PooledFlairEmbeddings,
    # RoBERTaEmbeddings,
    # TransformerXLEmbeddings,
    # WordEmbeddings,
    # XLMEmbeddings,
    # XLNetEmbeddings
)
# C.f https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md

app = Flask(__name__)

SIZE_EMBED = -1
adaptive_pool = nn.AdaptiveAvgPool1d(SIZE_EMBED) if SIZE_EMBED > 0 else None

embedder = BertEmbeddings(bert_model_or_path="distilbert-base-uncased",
                          pooling_operation="mean", use_scalar_mix=True)


@app.route('/vectorize',  methods=['POST'])
def vectorize():
    tokens = request.json["tokens"]

    embeddings = []
    # print("Call")
    for m in tokens:
        mot = Sentence(m)
        embedder.embed(mot)
        embed = mot[0].embedding

        if adaptive_pool is not None:
            embed = adaptive_pool(embed)

        embeddings.append(embed.detach().cpu().data.numpy().tolist())

    assert len(embeddings) == len(tokens)
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
    app.run(debug=True, threaded=True)


# @app.route('/vectorize',  methods=['POST'])
# def vectorize():
#     tokens = request.json["tokens"]
#     # print("\n\n\n", tokens, "\n\n\n")
#     embeddings = []
#     with torch.no_grad():
#         for tok in tokens:
#             encoded = bert_tok.encode(tok, return_tensors="pt")
#             out_tensor, hidden = bert(encoded)

#             out_tensor = adaptive_pool(out_tensor)
#             # hidden = adaptive_pool(hidden.unsqueeze(0))

#             # hidden = hidden.squeeze()
#             hidden = torch.sum(out_tensor, axis=1).squeeze()
#             assert len(hidden.size()) == 1
#             embeddings.append(hidden.detach().cpu().data.numpy().tolist())

#     # input_tensor = bert_tok.batch_encode_plus(tokens, pad_to_max_length=True,
#     #                                           return_tensors="pt")
#     # outputs = bert(input_tensor["input_ids"],
#     #                input_tensor["attention_mask"])
#     # outputs = adaptive_pool(outputs[0])
#     # outputs = outputs[0]
#     # embeddings = outputs.squeeze().cpu().data.numpy().tolist()
#     assert len(embeddings) == len(tokens)
#     for e in embeddings:
#         assert len(e) == 300
#     return jsonify({"vectors": embeddings})


# def vectorize_utterances():
#     utterances = request.json["utterances"]
#     with torch.no_grad():
#         input_tensor = bert_tok.batch_encode_plus(utterances,
#                                                   pad_to_max_length=True,
#                                                   return_tensors="pt")

#         outputs, hidden = bert(input_tensor["input_ids"],
#                                input_tensor["attention_mask"])

#         # outputs = adaptive_pool(outputs[0])
#         embedding = torch.sum(outputs, axis=1)
#         # embedding = adaptive_pool(hidden.unsqueeze(0)).squeeze()
#         # embedding = hidden
#         embeddings = embedding.detach().cpu().data.numpy().tolist()
#     return jsonify({"vectors": embeddings})
