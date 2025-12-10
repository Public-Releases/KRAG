import pandas as pd
import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.cluster import AgglomerativeClustering

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Preprocess text for matching"""
    if pd.isna(text) or not text:
        return ''
    text = re.sub(r'\(.*?\)', '', str(text)).strip()
    text = text.replace('_', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)

class DDXPlusKGRetriever:
    """KG retrieval adapted for DDXPlus dataset"""
    
    def __init__(self, kg_path='./dataset/llm_kg/knowledge_graph_DDXPlus_LLM_v2.xlsx'):
        """
        Initialize KG retriever
        
        Args:
            kg_path: Path to knowledge graph Excel file
        """
        self.kg_path = kg_path
        self.kg_data = None
        self.knowledge_graph = {}
        self.G = nx.Graph()
        self.symptom_nodes = []
        self.level_1_categories = set()
        self.level_2_categories = set()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.symptom_embeddings = None 
        
        self._load_kg()
        
        self.pipe = pipeline("image-text-to-text", model="Qwen/Qwen3-VL-2B-Instruct")
    
    def get_clense_prompt(self, symptom_text):
        return f"""
    Task: Extract specific medical symptoms and antecedents from the text.
    Rules:
    1. Output the summarized results in two or three words. No sentences. No explanations.
    2. Extract keywords ONLY. Do not infer diagnoses.

    Input: "Do you have a burning sensation that starts in your stomach then goes up into your throat?"
    Output: burning sensation in throat

    Input: "Are you significantly overweight?"
    Output: overweight
    
    Input: "Does the pain radiate to another location?: lower chest:
    Output: lower chest pain
    
    Input: {symptom_text}
    Output:
            """
    
    def clean_ddxplus_symptom(self, symptom_text):
        """
        Parses DDXPlus formatted symptoms into clinical keywords.
        """
        prompt = self.get_clense_prompt(symptom_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        response = self.pipe(text=messages)
        gen = response[0]["generated_text"]
        assistant_msg = gen[-1]["content"]
        if ',' in assistant_msg:
            return assistant_msg.split(',')
        else:
            return [assistant_msg]

    def semantic_deduplication_symptoms(self, symptoms, similarity_threshold=0.75):
        """
        Groups semantically similar symptoms and keeps the most concise version.
        
        Example: 
        Input: ['pain in head', 'headache', 'pain in the back of the head', 'fever']
        Output: ['headache', 'fever']
        
        Args:
            symptoms: List of symptoms
            similarity_threshold: Similarity threshold for clustering
        Returns:
            List of deduplicated symptoms
        """
        if not symptoms or len(symptoms) < 2:
            return symptoms
        # 2. Embed all symptoms
        embeddings = self.embedder.encode(symptoms, convert_to_tensor=False)
        
        # 3. Clustering
        distance_threshold = 1 - similarity_threshold
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        
        try:
            labels = clustering.fit_predict(embeddings)
        except ValueError:
            return list(set(symptoms))

        # 4. Consolidate
        deduplicated = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_items = [symptoms[i] for i in cluster_indices]
            
            shortest_term = min(cluster_items, key=len)
            deduplicated.append(shortest_term)
            
        return deduplicated
        

    def _load_kg(self):
        """Load and process knowledge graph"""
        print(f"Loading KG from: {self.kg_path}")
        
        # 1. Load Data
        self.kg_data = pd.read_excel(self.kg_path, usecols=['subject', 'relation', 'object'])
        
        self.knowledge_graph = {}
        self.G = nx.Graph()
        
        self.level_1_categories = set()
        self.level_2_categories = set()
        self.symptom_nodes = set()
        self.diseases = set()
        # 2. Iterate through every row
        for _, row in self.kg_data.iterrows():
            subject = str(row['subject']).strip()
            relation = str(row['relation']).strip()
            obj = str(row['object']).strip()
            
            # --- A. Build Dictionary (Adjacency List) ---
            if subject not in self.knowledge_graph:
                self.knowledge_graph[subject] = []
            self.knowledge_graph[subject].append((relation, obj))
            
            # --- B. Build NetworkX Graph ---
            self.G.add_edge(subject, obj, relation=relation)
            
            # --- C. Handle Specific Relations ---
            if relation == 'is_a':
                self.diseases.add(subject)
                if any(x in obj.strip().lower() for x in ['system', 'disease', 'disorder', 'infection']):
                    self.level_1_categories.add(obj)
                else:
                    self.level_2_categories.add(obj)

            elif relation in ['has_symptom', 'has_feature', 'presents_with', 
                              'has_symptomatology', 'has_anamnesis', 'has_lifestyle', 
                              'has_exposure', 'has_immunology', 'has_psychopathology']:
                
                self.diseases.add(subject)
                
                self.symptom_nodes.add(obj)

            elif relation in ['similar_to', 'distinguished_by']:
                self.diseases.add(subject)
                
            elif relation in ['risk_factor', 'associated_with']:
                self.diseases.add(subject)
                self.symptom_nodes.add(obj)

        self.symptom_nodes = list(self.symptom_nodes)
        
        print(f"Loaded KG: {len(self.kg_data)} triples")
        print(f"  - Level 1 categories: {len(self.level_1_categories)}")
        print(f"  - Level 2 categories: {len(self.level_2_categories)}")
        print(f"  - Total Diseases found: {len(self.diseases)}")
        print(f"  - Symptom/Feature nodes: {len(self.symptom_nodes)}")
    
    def get_disease_for_level2_category(self, level2_category):
        """
        Get diseases that belong to a Level 2 category
        
        Args:
            level2_category: Level 2 category name
            
        Returns:
            List of diseases in that category
        """
        diseases = []
        
        # Find all subjects that have is_a relationship to this category
        for subject, edges in self.knowledge_graph.items():
            for relation, obj in edges:
                if relation == 'is_a' and obj == level2_category:
                    diseases.append(subject)
        
        return diseases
    
    def get_symptoms_for_disease(self, disease_name):
        """Get all symptoms associated with a disease"""
        symptoms = []
        
        if disease_name in self.knowledge_graph:
            for relation, obj in self.knowledge_graph[disease_name]:
                if relation in ['has_symptom', 'has_feature', 'distinguished_by']:
                    symptoms.append(obj)
        
        return symptoms
    
    def _precompute_kg_embeddings(self):
        """Call this once after loading KG"""
        # Encode all KG symptom nodes once
        self.symptom_embeddings = self.embedder.encode(self.symptom_nodes, convert_to_tensor=True)

    def get_relevant_diseases_from_symptoms(self, patient_symptoms, top_n=3):
        """
        Find most relevant diseases based on semantic similarity
        """
        if self.symptom_embeddings is None:
            self._precompute_kg_embeddings()

        disease_scores = {}
        
        clean_symptoms = []
        for s in patient_symptoms:
            clean_symptom = self.clean_ddxplus_symptom(s)
            if clean_symptom:
                if isinstance(clean_symptom, list):
                    clean_symptoms.extend(clean_symptom)
                else:
                    clean_symptoms.append(clean_symptom)
        
        # Semantic deduplication of clean symptoms
        clean_symptoms = self.semantic_deduplication_symptoms(clean_symptoms)
        
        noisy_keywords = ['reason for consulting', 'intensity', 'unspecified', 'appeared fast', 'scale', 'related to']

        clean_symptoms = [
            s for s in clean_symptoms 
            if not any(noise in s.lower() for noise in noisy_keywords)
        ]
        
        if not clean_symptoms:
            print(f'No clean symptoms found for patient symptoms')
            return []

        patient_embeddings = self.embedder.encode(clean_symptoms, convert_to_tensor=True)

        hits = util.semantic_search(patient_embeddings, self.symptom_embeddings, top_k=1)

        # Aggregate Scores
        for i, hit in enumerate(hits):
            best_match = hit[0]
            score = best_match['score']
            kg_idx = best_match['corpus_id']
            
            if score > 0.4:
                matched_kg_symptom = self.symptom_nodes[kg_idx]
                
                diseases = self._get_diseases_with_symptom(matched_kg_symptom)
                for disease in diseases:
                    # Add the similarity score to the disease ranking
                    disease_scores[disease] = disease_scores.get(disease, 0) + score

        # Sort by accumulated score
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_diseases[:top_n], clean_symptoms, None
    
    def get_relevant_diseases_from_symptoms_OLD(self, patient_symptoms, top_n=3):
        """
        MedRAG source code retrieval method
        """
        if self.symptom_embeddings is None:
            self._precompute_kg_embeddings()
        
        disease_scores = {}
        patient_embeddings = self.embedder.encode(patient_symptoms, convert_to_tensor=True)

        hits = util.semantic_search(patient_embeddings, self.symptom_embeddings, top_k=1)

        for i, hit in enumerate(hits):
            best_match = hit[0]
            score = best_match['score']
            kg_idx = best_match['corpus_id']
            
            if score > 0.5:
                matched_kg_symptom = self.symptom_nodes[kg_idx]
                
                diseases = self._get_diseases_with_symptom(matched_kg_symptom)
                for disease in diseases:
                    disease_scores[disease] = disease_scores.get(disease, 0) + score

        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_diseases[:top_n]
    
    def _get_diseases_with_symptom(self, symptom):
        """Find diseases that have a specific symptom"""
        diseases = []
        
        for _, row in self.kg_data.iterrows():
            if row['object'] == symptom and row['relation'] in ['has_symptom', 'has_feature', 'has_symptomatology', 'has_exposure', 'has_anamnesis']: # add more relations here if needed
                diseases.append(row['subject'])
        
        return diseases
    
    def get_additional_context_for_disease(self, disease_name):
        """
        Get comprehensive information about a disease from KG
        
        Returns a formatted string with:
        - Critical symptoms
        - Diagnostic features  
        - Risk factors
        - Similar diseases
        """
        context_parts = []
        
        if disease_name not in self.knowledge_graph:
            return f"No KG information found for {disease_name}"
        
        symptoms = []
        features = []
        risk_factors = []
        similar = []
        
        for relation, obj in self.knowledge_graph[disease_name]:
            if relation == 'has_symptom':
                symptoms.append(obj)
            elif relation == 'has_feature':
                features.append(obj)
            elif relation == 'risk_factor':
                risk_factors.append(obj)
            elif relation == 'similar_to':
                similar.append(obj)
        
        if symptoms:
            context_parts.append(f"Key symptoms: {', '.join(symptoms[:5])}")
        if features:
            context_parts.append(f"Diagnostic features: {', '.join(features[:3])}")
        if risk_factors:
            context_parts.append(f"Risk factors: {', '.join(risk_factors[:3])}")
        if similar:
            context_parts.append(f"Consider differential: {', '.join(similar[:3])}")
        
        return " | ".join(context_parts) if context_parts else f"Limited information for {disease_name}"
    
    def _get_level2_category_for_disease(self, disease):
            if disease not in self.knowledge_graph:
                return None
                
            for relation, obj in self.knowledge_graph[disease]:
                if relation == 'is_a':
                    if obj in self.level_2_categories:
                        return obj
                    if obj in self.level_1_categories:
                        return obj
            return None
        
    def get_relevant_diseases_hierarchical(self, patient_symptoms, top_n=3):
        """
        Reproduction of MedRAG KG retrieval
        1. Match symptoms to KG nodes.
        2. 'Vote' for the Level 2 Category (e.g. Respiratory, Gastro) that these nodes belong to.
        3. Select the winning Category.
        4. Retrieve and rank ONLY diseases within that Category.
        """
        if self.symptom_embeddings is None:
            self._precompute_kg_embeddings()

        clean_symptoms = []
        for s in patient_symptoms:
            clean = self.clean_ddxplus_symptom(s)
            if isinstance(clean, list):
                clean_symptoms.extend(clean)
            else:
                clean_symptoms.append(clean)
        
        clean_symptoms = self.semantic_deduplication_symptoms(clean_symptoms)
        
        if not clean_symptoms:
            return [], []

        patient_embeddings = self.embedder.encode(clean_symptoms, convert_to_tensor=True)
        
        hits = util.semantic_search(patient_embeddings, self.symptom_embeddings, top_k=3)
        
        category_votes = {}
        matched_disease_scores = {}

        for i, hit_list in enumerate(hits):
            for hit in hit_list:
                score = hit['score']
                if score < 0.4: continue
                
                kg_idx = hit['corpus_id']
                matched_node = self.symptom_nodes[kg_idx]
                
                connected_diseases = self._get_diseases_with_symptom(matched_node)
                
                for disease in connected_diseases:
                    if disease in ['Localized edema', 'Anemia']: continue

                    matched_disease_scores[disease] = matched_disease_scores.get(disease, 0) + score
                    
                    parent_cat = self._get_level2_category_for_disease(disease)
                    
                    if parent_cat:
                        category_votes[parent_cat] = category_votes.get(parent_cat, 0) + score

        if not category_votes:
            return [], clean_symptoms
        winning_category = max(category_votes, key=category_votes.get)
        print(f"Hierarchical Voting Winner: {winning_category} (Score: {category_votes[winning_category]:.2f})")
        
        candidate_diseases = self.get_disease_for_level2_category(winning_category)
        
        final_ranking = []
        for disease in candidate_diseases:
            score = matched_disease_scores.get(disease, 0)
            if score > 0:
                final_ranking.append((disease, score))
        
        final_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return final_ranking[:top_n], clean_symptoms, winning_category 

def get_additional_info_ddxplus(kg_retriever, patient_case, top_n=3):
    """
    Get additional diagnostic context from KG for DDXPlus
    
    Args:
        kg_retriever: DDXPlusKGRetriever instance
        patient_case: Patient case dict or JSON string
        top_n: Number of top diseases to consider
        
    Returns:
        String with additional diagnostic context
    """
    import json
    
    if isinstance(patient_case, str):
        try:
            patient_case = json.loads(patient_case)
        except:
            return "No additional information available.", patient_symptoms, None
    
    patient_symptoms = []
    
    if 'Symptoms' in patient_case:
        symptoms = patient_case['Symptoms']
        if isinstance(symptoms, list):
            patient_symptoms.extend(symptoms)
        elif isinstance(symptoms, str):
            patient_symptoms.append(symptoms)
    
    if 'Antecedents' in patient_case:
        antecedents = patient_case['Antecedents']
        if isinstance(antecedents, list):
            patient_symptoms.extend(antecedents)
    
    if not patient_symptoms:
        return "No symptoms found in patient case.", patient_symptoms, None
    
    # Get relevant diseases from KG
    if 'LLM' in kg_retriever.kg_path:
        relevant_diseases, clean_symptoms, winning_category = kg_retriever.get_relevant_diseases_from_symptoms(patient_symptoms, top_n=top_n)
        # relevant_diseases, clean_symptoms, winning_category = kg_retriever.get_relevant_diseases_hierarchical(patient_symptoms, top_n=top_n)
    else:
        print(f'Baseline KG detected.')
        relevant_diseases = kg_retriever.get_relevant_diseases_from_symptoms_OLD(patient_symptoms, top_n=top_n)
        clean_symptoms = patient_symptoms
        winning_category = None
    
    if not relevant_diseases:
        return "No matching diseases found in knowledge graph.", patient_symptoms, None
    
    context_parts = []
    context_parts.append(f"Based on symptoms, consider these diagnoses:")
    
    for disease, score in relevant_diseases[:top_n]:
        disease_context = kg_retriever.get_additional_context_for_disease(disease)
        context_parts.append(f"\n{disease} (relevance: {score}): {disease_context}")
    
    return "\n".join(context_parts), clean_symptoms, winning_category