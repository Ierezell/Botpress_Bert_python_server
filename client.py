import requests
import json
import numpy as np

r = requests.post("http://127.0.0.1:5000/vectorize",
                  #   data={"token": json.dumps("Coucou toi ! Je t'aime")},
                  json={"sentence": "Coucou toi ! Je t'aime"}
                  )
print(r.json())
print(type(r.json()))
t = np.array(r.json())
print(t.shape)
print(len(r.json()[0]))
