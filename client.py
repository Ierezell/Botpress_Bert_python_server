import requests
import json
import numpy as np

r = requests.post("http://127.0.0.1:5000/vectorize_utterances",
                  json={"utterances": [
                      "Coucou toi ! Je t'aime",
                      "Bonjour, on fait du typescript ! C'est génial."],
                      "lang": "fr"}
                  )
print(r)
print(r.json())
# print(type(r.json()))
# t = np.array(r.json())
# print(t.shape)
# print(len(r.json()[0]))
