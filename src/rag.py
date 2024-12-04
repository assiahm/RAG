import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer



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

# 3. Création d'un index vectoriel :
# Chargement du modèle d'embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Génération des embeddings
embeddings = [model.encode(chunk) for chunk in chunks]
print(f"Nombre d'embeddings générés : {len(embeddings)}")

# 4. Création d’une base de données vectorielle:
# Nouvelle configuration pour la base de données
client = chromadb.Client(Settings(persist_directory="db"))

# Création d'une collection pour stocker les embeddings
collection = client.create_collection("product_descriptions")

# Ajout des embeddings à la collection
for i, embedding in enumerate(embeddings):
    collection.add( 
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        metadatas=[{"text": chunks[i]}]
    )
print("Base de données vectorielle créée et remplie.")


# 5. Création d’un système de récupération (retrieval system)
def retrieve_documents(query):
    # Génération de l'embedding pour la requête
    query_embedding = model.encode(query)
    
    # Requête dans la base de données vectorielle
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Nombre de résultats à récupérer
    )
    
    # Extraction des textes pertinents
    return results["metadatas"]


# Exemple de test pour afficher la structure
query = "Tell me about OnePlus 6T"
retrieved_docs = retrieve_documents(query)

print("Structure des résultats :")
print(retrieved_docs)  # Inspecter les résultats avant de les parcourir

"""
# Affichage des documents pertinents
if isinstance(retrieved_docs, list):
    print("Documents pertinents récupérés :")
    for doc in retrieved_docs:
        if isinstance(doc, dict) and "text" in doc:
            print(doc["text"])
        else:
            print("Document non conforme :", doc)
else:
    print("Aucun résultat ou format inattendu :", retrieved_docs)

"""
#6. Réglage du LLM
def create_prompt(query, documents):
    # Combine les documents récupérés dans un seul bloc
    context = "\n".join([doc["text"] for doc in documents])
    prompt = f"""
    Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement
    les informations fournies ci-dessous :

    {context}

    Question : {query}

    Répondez précisément et de manière concise en utilisant uniquement les informations fournies. 
    Si vous ne trouvez pas la réponse, indiquez "Je ne sais pas". 
    Ne faites pas appel à des connaissances externes ou internes et fournissez toujours les passages 
    ou documents spécifiques utilisés pour répondre.
    """
    return prompt
