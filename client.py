import requests
import json
import numpy as np

r = requests.post("http://127.0.0.1:5000/vectorize_utterances",
                  json={"utterances": ["Hey my friend where is the dog at ?", "My cousin is a saxo artist"],
                  "lang": "en"})

print(np.array(r.json()['vectors']).shape)

r = requests.post("http://127.0.0.1:5000/vectorize_utterances",
                  json={"utterances": ["Is this real ?"],
                  "lang": "en"})

print(np.array(r.json()['vectors']).shape)