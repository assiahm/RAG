import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Désactiver le parallélisme des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Charger les données et préparer les embeddings
@st.cache_resource
def load_and_prepare_data(file_path, model_name="all-MiniLM-L12-v2"):
    with open(file_path, 'r', encoding='utf-8') as file:
        product_data = [json.loads(line) for line in file]
    
    descriptions = [product.get("description", "") for product in product_data if "description" in product]
    
    # Segmentation des textes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    segmented_texts = []
    for description in descriptions:
        if isinstance(description, list):
            description = " ".join(description)
        if description:
            chunks = text_splitter.split_text(description)
            segmented_texts.extend(chunks)
    
    # Générer des embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embedding_model.embed_documents(segmented_texts)
    
    # Créer un index FAISS
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Créer un docstore
    documents = [Document(page_content=text) for text in segmented_texts]
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    
    # Créer un vector store
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vector_store

# Charger un modèle LLM (comme BLOOM) pour générer des réponses
@st.cache_resource
def load_llm_model(model_name="bigscience/bloomz-560m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Extraire et vérifier uniquement les informations pertinentes
def get_flexible_answer(query, retriever, max_results=3):
    results = retriever.get_relevant_documents(query)
    
    if not results:
        return "Je ne sais pas.", []  # No documents found

    # Score documents for relevance based on overlapping words
    matched_contexts = []
    query_words = set(query.lower().split())
    for result in results[:max_results]:
        document_words = set(result.page_content.lower().split())
        overlap_score = len(query_words & document_words)
        if overlap_score > 0:  # Ensure some relevance
            matched_contexts.append((result.page_content, overlap_score))
    
    if not matched_contexts:
        return "Je ne sais pas.", []  # No directly relevant documents
    
    # Sort by relevance (overlap score) and return the most relevant
    matched_contexts = sorted(matched_contexts, key=lambda x: x[1], reverse=True)
    filtered_contexts = [context[0] for context in matched_contexts]

    return "\n\n".join(filtered_contexts[:max_results]).strip(), filtered_contexts

# Générer une réponse concise avec LLM
def generate_response(query, context, tokenizer, model, temperature, top_p):
    # Truncate context to avoid token limits
    truncated_context = context[:1000]

    # Enhanced prompt with clear instructions and structured output
    prompt = f"""
    Vous êtes un assistant expert en descriptions de produits. Répondez précisément à la question suivante en utilisant les informations ci-dessous. N'utilisez aucune connaissance externe.

    Contexte fourni :
    {truncated_context}

    Instructions :
    - Répondez de manière concise et factuelle.
    - Si une réponse ne peut pas être trouvée dans le contexte, répondez strictement : "Je ne sais pas."
    - Formatez la réponse de manière claire, et indiquez les informations pertinentes en citant les passages correspondants si possible.

    Question : {query}
    Réponse :
    """

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

#def clean_response(response, query):
   # """
    #Supprime uniquement les parties non désirées tout en préservant la réponse générée.
   # """
    # Supprimer les instructions initiales si présentes
    #if "Répondez directement" in response:
        #response = response.split("Répondez directement", maxsplit=1)[-1]
    
    # Retirer les instructions résiduelles
   # if "Si aucune réponse ne peut être trouvée" in response:
        #response = response.split("Si aucune réponse ne peut être trouvée")[0]

    # Nettoyage final
   # return response.strip()

def test_parameters(query, retriever, tokenizer, model, temp_values, top_p_values):
    results = []
    context, _ = get_flexible_answer(query, retriever)

    for temp in temp_values:
        for top_p in top_p_values:
            response = generate_response(query, context, tokenizer, model, temp, top_p)
            results.append({
                "query": query,
                "temperature": temp,
                "top_p": top_p,
                "response": response
            })
    return results

# Streamlit main function
def main():
    st.set_page_config(page_title="Descriptions Produits Amazon", layout="wide")
    st.title("🔍Descriptions De Produits Amazon")
    st.write("Posez une question pour obtenir des réponses basées sur les descriptions de produits fournies.")
    
    file_path = 'data/meta.jsonl'  # Path to your JSONL file
    vector_store = load_and_prepare_data(file_path)
    retriever = vector_store.as_retriever()

    tokenizer, model = load_llm_model()

    # Sidebar for parameters
    with st.sidebar:
        st.header("🔧 Paramètres")
        temperature = st.slider("Température (créativité)", 0.1, 2.0, 1.0, step=0.1)
        top_p = st.slider("Top-p (probabilité cumulative)", 0.1, 1.0, 0.9, step=0.1)
    
    # User input
    query = st.text_input("💬 Posez votre question :")

    if st.button("🚀 Obtenir une réponse"):
        if query.strip():
            # Retrieve relevant context
            context, matched_contexts = get_flexible_answer(query, retriever)
            
            if context == "Je ne sais pas.":
                st.write("### Réponse :")
                st.write(context)
            else:
                # Display relevant documents
                if matched_contexts:
                    st.write("### Documents Pertinents :")
                    for idx, doc in enumerate(matched_contexts):
                        st.write(f"{idx + 1}. {doc[:300]}...")  # Display first 300 characters

                # Generate a concise response
                response = generate_response(query, context, tokenizer, model, temperature, top_p)
                st.write("### Réponse :")
                st.write(response)
                
                # Log parameters used
                st.write("#### Paramètres utilisés :")
                st.write(f"Température : {temperature}, Top-p : {top_p}")
        else:
            st.write("Veuillez poser une question valide.")
    
    # Test parameters
    if st.button("🧪 Tester les paramètres"):
        temp_values = [0.5, 1.0, 1.5]
        top_p_values = [0.8, 0.9, 1.0]
        test_results = test_parameters(query, retriever, tokenizer, model, temp_values, top_p_values)
        
        st.write("### Résultats des tests avec différents paramètres :")
        for result in test_results:
            st.write(f"- **Température** : {result['temperature']}, **Top-p** : {result['top_p']}")
            st.write(f"**Réponse** : {result['response']}")

if __name__ == "__main__":
    main()
