import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Chargement des données
with open('data/meta.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# Transformation des descriptions en chaînes de caractères
descriptions = []
for item in data:
    desc = item.get("description")
    if isinstance(desc, list):  # Si la description est une liste, on la convertit en chaîne
        descriptions.append(" ".join(desc))
    elif isinstance(desc, str):  # Si la description est déjà une chaîne
        descriptions.append(desc)

# 2. Prétraitement et segmentation des textes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = []

# Découpage des descriptions en morceaux
for desc in descriptions:
    if desc.strip():  # Vérifie que la description n'est pas vide après traitement
        chunks.extend(text_splitter.split_text(desc))

# Résultats
print(f"Nombre total de descriptions valides : {len(descriptions)}")
print(f"Nombre total de chunks générés : {len(chunks)}")

#3.Création d'un index vectoriel :

from sentence_transformers import SentenceTransformer

# Chargement du modèle d'embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Génération des embeddings
embeddings = [model.encode(chunk) for chunk in chunks]
print(f"Nombre d'embeddings générés : {len(embeddings)}")
