import json

with open('data/meta.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]
descriptions = [item["description"] for item in data if "description" in item]

print(descriptions)