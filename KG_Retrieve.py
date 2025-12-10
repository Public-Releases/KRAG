"""
So far unusable code for any dataset

"""
import openai
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

# Import API key from authentication.py
try:
    from authentication import api_key
except ImportError:
    api_key = ''
    print("Warning: Could not import API key from authentication.py")

# client = openai.OpenAI(api_key=api_key)
client = None

# Default paths - can be overridden for different datasets
# KG_file_path = './dataset/knowledge graph of chronic pain.xlsx'
KG_file_path = './dataset/medrag_kg_ddxplus.xlsx'
# Try DDXPlus knowledge graph if chronic pain one doesn't exist
# if not os.path.exists(KG_file_path):
#     ddxplus_kg_path = './dataset/knowledge graph of DDXPlus.xlsx'
#     if os.path.exists(ddxplus_kg_path):
#         KG_file_path = ddxplus_kg_path

file_path = './dataset/AI Data Set with Categories.csv'
embedding_save_path = './Embeddings_saved/CP_KG_embeddings'



def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = text.replace('_', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)


# Load knowledge graph if the file exists
knowledge_graph = {}
kg_data = None
symptom_nodes = []

if os.path.exists(KG_file_path):
    kg_data = pd.read_excel(KG_file_path, usecols=['subject', 'relation', 'object'])
    
    for index, row in kg_data.iterrows():
        subject = row['subject']
        relation = row['relation']
        obj = row['object']

        if subject not in knowledge_graph:
            knowledge_graph[subject] = []
        knowledge_graph[subject].append((relation, obj))
        
        # Also add reverse relationships for bidirectional graph
        if obj not in knowledge_graph:
            knowledge_graph[obj] = []
        knowledge_graph[obj].append((relation, subject))
    
    # Preprocess symptoms for matching
    kg_data['object_preprocessed'] = kg_data.apply(
        lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
        axis=1
    )
    symptom_nodes = kg_data['object_preprocessed'].dropna().unique().tolist()
else:
    print(f"Warning: Knowledge graph file not found at {KG_file_path}")

def get_symptom_embeddings(symptom_nodes, save_path, use_local_embeddings=True):
    embeddings_path = os.path.join(save_path, 'KG_embeddings.npy')
    if os.path.exists(embeddings_path):
        print("load existing embeddings...")
        return np.load(embeddings_path)
    else:
        if use_local_embeddings:
            print("generate new embeddings using local sentence-transformers model (FREE)...")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                symptom_embeddings = model.encode(symptom_nodes, show_progress_bar=False)
                np.save(embeddings_path, symptom_embeddings)
                return np.array(symptom_embeddings)
            except ImportError:
                print("sentence-transformers not installed. Installing...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'sentence-transformers'])
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                symptom_embeddings = model.encode(symptom_nodes, show_progress_bar=False)
                np.save(embeddings_path, symptom_embeddings)
                return np.array(symptom_embeddings)
        else:
            print("generate new embeddings using OpenAI API (requires API key and costs money)...")
            symptom_embeddings = []
            for symptom in symptom_nodes:
                response = client.embeddings.create(
                    input=symptom,
                    model="text-embedding-3-large"
                )
                symptom_embeddings.append(response.data[0].embedding)
            np.save(embeddings_path, symptom_embeddings)
            return np.array(symptom_embeddings)


symptom_embeddings = None

def load_symptom_embeddings():
    """Lazy loading of symptom embeddings to avoid OpenAI API call at import time"""
    global symptom_embeddings
    if symptom_embeddings is None:
        symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path, use_local_embeddings=True)
    return symptom_embeddings


_sentence_transformer_model = None

def get_query_embedding_local(query):
    """Get embedding for a query using local sentence-transformers model"""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_transformer_model.encode([query])[0]


def find_top_n_similar_symptoms(query, symptom_nodes, symptom_embeddings, n, use_local_embeddings=True):
    if pd.isna(query) or not query:
        return []
    query_preprocessed = preprocess_text(query)
    
    if use_local_embeddings:
        query_embedding = get_query_embedding_local(query_preprocessed)
    else:
        response = client.embeddings.create(
            input=query_preprocessed,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
    if not query_embedding:
        return []

    if len(symptom_embeddings) > len(symptom_nodes):
        symptom_embeddings = symptom_embeddings[:len(symptom_nodes)]

    similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

    top_n_symptoms = []
    unique_symptoms = set()
    top_n_indices = similarities.argsort()[::-1]

    for i in top_n_indices:
        if similarities[i] > 0.5 and symptom_nodes[i] not in unique_symptoms:
            top_n_symptoms.append(symptom_nodes[i])
            unique_symptoms.add(symptom_nodes[i])
        if len(top_n_symptoms) == n:
            break

    return top_n_symptoms


def compute_shortest_path_length(node1, node2, G):
    try:
        return nx.shortest_path_length(G, source=node1, target=node2)
    except nx.NetworkXNoPath:
        return float('inf')

categories = [
    "thoracoabdominal_pain_syndromes",
    "neuropathic_pain_syndromes",
    "craniofacial_pain_syndromes",
    "cervical_spine_pain_syndromes",
    "limb_and_joint_pain_syndromes",
    "back_pain_syndromes",
    "lumbar_degenerative_and_stenosis_and_radicular_and_sciatic_syndromes",
    "generalized_pain_syndromes",

]
G = nx.Graph()
for node, edges in knowledge_graph.items():
    for relation, neighbor in edges:
        G.add_edge(node, neighbor, relation=relation)


def get_diagnoses_for_symptom(symptom):
    diagnoses = []
    if symptom in G:
        for neighbor in G.neighbors(symptom):
            edge_data = G.get_edge_data(neighbor, symptom)
            if edge_data and 'relation' in edge_data and edge_data['relation'] != 'is_a':
                diagnoses.append(neighbor)
    return diagnoses


def find_closest_category(top_symptoms, categories,top_n):
    if isinstance(top_symptoms, pd.Series) and top_symptoms.empty:
        print("Warning: top_symptoms is empty.")
        return None
    category_votes = {category: 0 for category in categories}
    for symptom in top_symptoms:
        top_symptoms = list(set(top_symptoms))

        # print('symptom: ',symptom)
        if symptom not in G:
            print(f"Symptom node not found in graph: {symptom}")
            continue

        diagnosis_nodes = get_diagnoses_for_symptom(symptom)
        for diagnosis in diagnosis_nodes:

            individual_diagnoses = diagnosis.split(',')

            for single_diagnosis in individual_diagnoses:
                single_diagnosis = single_diagnosis.strip().replace(' ', '_').lower()  # 去掉前后空格
                if single_diagnosis not in G:
                    print(f"Diagnosis node not found in graph: {single_diagnosis}")
                    continue

                min_distance = float('inf')
                closest_category = None

                for category in categories:
                    if category not in G:
                        print(f"Category node not found in graph: {category}")
                        continue

                    try:
                        distance = nx.shortest_path_length(G, source=single_diagnosis, target=category)
                    except nx.NetworkXNoPath:
                        distance = float('inf')

                    if distance < min_distance:
                        min_distance = distance
                        closest_category = category

                if closest_category:
                    category_votes[closest_category] += 1
    print("Category votes:", category_votes)

    sorted_categories = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)
    top_n_categories = [sorted_categories[i][0] for i in range(top_n)]
    return top_n_categories


def get_keyinfo_for_category(category, knowledge_graph):
    keyinfo_values = []
    for node, edges in knowledge_graph.items():
        if node == category:
            for relation, neighbor in edges:
                if relation == "is_a" and neighbor in knowledge_graph:
                    for rel, obj in knowledge_graph[neighbor]:
                        if rel == "has_keyinfo":
                            keyinfo_values.append(obj)
    return keyinfo_values



def get_subjects_for_objects(objects, knowledge_graph):
    subjects = []
    processed_objects = [obj.replace(' ', '_') for obj in objects]
    for obj in processed_objects:
        for index, row in knowledge_graph.iterrows():
            if row['object'] == obj:
                subjects.append(row['subject'])
    return subjects


def find_level3_for_symptoms(top_symptoms, knowledge_graph):
    level3_connections = {}
    for symptom in top_symptoms:
        subjects = get_subjects_for_objects([symptom], knowledge_graph)
        for subject in subjects:
            if subject in level3_connections:
                level3_connections[subject] += 1
            else:
                level3_connections[subject] = 1
    return level3_connections


def print_symptom_and_disease(symptom_nodes):
    for symptom in symptom_nodes:
        subjects = get_subjects_for_objects([symptom], kg_data)


def main_get_category_and_level3(n, participant_no,top_n):
    global symptom_embeddings
    symptom_embeddings = load_symptom_embeddings()
    
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    row = data.loc[data['Participant No.'] == int(participant_no)]
    if row.empty:
        print(f"Participant No. {participant_no} not found!")
        return None

    tr = row["Level 2"].values[0] if "Level 2" in row.columns else "Unknown"
    tr = tr.split(",")[0] if isinstance(tr, str) else str(tr)

    level3real = row["Processed Diagnosis"].values[0]

    pain_location = row["Pain Presentation and Description"].values[0] if "Pain Presentation and Description" in row.columns else ''
    pain_symptoms = row["Pain descriptions and assorted symptoms (self-report)"].values[0] if "Pain descriptions and assorted symptoms (self-report)" in row.columns else ''
    pain_restriction = row["Pain restriction"].values[0] if "Pain restriction" in row.columns else ''
    
    if not pain_symptoms and "Symptoms" in row.columns:
        pain_symptoms = row["Symptoms"].values[0]
    
    print(f'pain_location: {pain_location}')
    print(f'pain_symptoms: {pain_symptoms}')
    print(f'pain_restriction: {pain_restriction}')
    
    if pd.isna(pain_location):
        pain_location = ''
    if pd.isna(pain_symptoms):
        pain_symptoms = ''
    if pd.isna(pain_restriction):
        pain_restriction = ''


    def process_symptom_field(field_value, symptom_nodes, symptom_embeddings, n):
        if pd.isna(field_value) or field_value == '':
            return []
        return find_top_n_similar_symptoms(field_value, symptom_nodes, symptom_embeddings, n)

    top_5_location_nodes = process_symptom_field(pain_location, symptom_nodes, symptom_embeddings, n)
    top_5_symptom_nodes = process_symptom_field(pain_symptoms, symptom_nodes, symptom_embeddings, n)
    top_5_painrestriction_nodes = process_symptom_field(pain_restriction, symptom_nodes, symptom_embeddings, n)


    top_5_location_nodes_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_location_nodes), 'object'].drop_duplicates()
    top_5_symptom_nodes_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_symptom_nodes), 'object'].drop_duplicates()
    top_5_painrestriction_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_painrestriction_nodes), 'object'].drop_duplicates()


    most_similar_category = find_closest_category(
        list(top_5_location_nodes_original) + list(top_5_symptom_nodes_original)+ list(top_5_painrestriction_original),
        categories,
        top_n
    )
    return most_similar_category
