import json

data = {
    "image": [0.0] * 784,
    "width": 28,
    "height": 28
}

with open("payload.json", "w") as f:
    json.dump(data, f)
