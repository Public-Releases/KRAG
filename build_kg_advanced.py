import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import os
import regex as re
from local_llm_inference import generate_diagnosis_with_local_llm
from transformers import pipeline
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, util

# Try to import Gemini support
try:
    from gemini_integration import generate_with_gemini, generate_json_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class LLMKGBuilder:
    """Build knowledge graph using LLM as medical expert"""
    
    def __init__(self, model='local/llama-3.1-8b', output_dir='./dataset/llm_kg', use_gemini=False):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gemini = use_gemini
        
        # Map model shortcuts to actual paths
        self.model_map = {
            'local/llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'local/openbiollm-8b': './models/llama3-openbiollm-8b',
            'local/llama-3.1-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'gemini-1.5-pro': 'gemini-1.5-pro-latest',
            'gemini-1.5-flash': 'gemini-1.5-flash-latest',
            'gemini-2.0-flash': 'gemini-2.0-flash',           # 2,000 RPM
            'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite', # 4,000 RPM
            'gemini-2.5-flash': 'gemini-2.5-flash',           # 1,000 RPM (balanced)
            'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite', # 4,000 RPM (fastest)
            'gemini-2.5-pro': 'gemini-2.5-pro',               # 150 RPM (best quality)
        }
        
        self.hf_model = self.model_map.get(model, model)
        
        if self.use_gemini and not GEMINI_AVAILABLE:
            self.use_gemini = False
            
        self.ddxplus_dir = './dataset/df/train'
        self.ddxplus_files = [f for f in os.listdir(self.ddxplus_dir) if f.endswith('.json')]
        
        self.pipe = pipeline("image-text-to-text", model="Qwen/Qwen3-VL-2B-Instruct")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_diseases(self):
        """Load all diseases from DDXPlus conditions file"""
        conditions_file = './DDXPlus/release_conditions.json'
        
        with open(conditions_file, 'r') as f:
            conditions = json.load(f)
        
        diseases = []
        if isinstance(conditions, dict):
            for disease_name, disease_info in conditions.items():
                diseases.append({
                    'name': disease_info.get('condition_name', disease_name),
                    'icd10': disease_info.get('icd10-id', ''),
                    'severity': disease_info.get('severity', ''),
                    'cond_name_eng': disease_info.get('cond-name-eng', disease_name)
                })
        
        return diseases
    
    def cluster_diseases(self, diseases, resume=True):
        """Step 1: Cluster diseases using LLM"""
        
        checkpoint_file = self.output_dir / 'taxonomy_checkpoint.json'
        
        # Check if we can resume
        if resume and checkpoint_file.exists():
            print(f"✓ Found existing taxonomy at {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        
        disease_names = [d['name'] for d in diseases]
        
        system_prompt = """You are a medical taxonomy API. Return ONLY valid JSON. Use actual disease names as keys."""

        user_prompt = f"""Create a medical taxonomy for these {len(disease_names)} diseases.

DISEASES:
{chr(10).join(f'{i+1}. {name}' for i, name in enumerate(disease_names))}

For EACH disease above, create an entry with:
- level_1: Organ system (e.g., "Respiratory System", "Cardiovascular System")
- level_2: Subcategory (e.g., "Upper Respiratory Infections", "Ischemic Heart Disease")
- level_3: The exact disease name from the list above
- similar_diseases: 2-3 diseases from the list that are clinically similar

CRITICAL REQUIREMENTS:
1. Use the EXACT disease names from the list above as JSON keys
2. Return valid JSON only, no explanations
3. Include ALL {len(disease_names)} diseases
4. Start with {{ and end with }}

Example format (use actual names, not "Disease1"):
{{
  "Pneumonia": {{
    "level_1": "Respiratory System",
    "level_2": "Lower Respiratory Infections",
    "level_3": "Pneumonia",
    "similar_diseases": ["Bronchitis", "Influenza"]
  }},
  "Myocardial infarction": {{
    "level_1": "Cardiovascular System",
    "level_2": "Ischemic Heart Disease",
    "level_3": "Myocardial infarction",
    "similar_diseases": ["Unstable angina", "Stable angina"]
  }}
}}"""

        print("\n" + "="*80)
        print("STEP 1: LLM Disease Clustering")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Clustering {len(disease_names)} diseases...")
        
        # Use Gemini if available and requested
        if self.use_gemini:
            print("Using Gemini API for taxonomy generation...")
            from authentication import google_api_key
            
            try:
                taxonomy = generate_json_with_gemini(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=google_api_key,
                    model_name=self.hf_model,
                    temperature=0.2,
                    max_tokens=8192
                )
                
                if taxonomy:
                    print(f"Successfully generated taxonomy for {len(taxonomy)} diseases using Gemini!")
                else:
                    print("Gemini returned None")
                    
            except Exception as e:
                print(f"Gemini API error: {e}")
                taxonomy = None
        else:
            # Use local LLM
            print("Using local LLM...")
            response = generate_diagnosis_with_local_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.hf_model,
                max_new_tokens=6000
            )
            
            # Parse response
            taxonomy = self._parse_json_response(response)
            
            if taxonomy:
                print(f"Parsed {len(taxonomy)}/{len(disease_names)} diseases")
            else:
                print("Failed to parse taxonomy")        
        if taxonomy:
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(taxonomy, f, indent=2)
            print(f"Taxonomy saved to {checkpoint_file}")
        
        return taxonomy
    
    def generate_disease_features(self, diseases, resume=True, batch_size=5):
        """Step 2: Generate features for each disease using LLM"""
        
        checkpoint_file = self.output_dir / 'features_checkpoint.json'
        
        # Load existing progress if resuming
        if resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                disease_features = json.load(f)
            print(f"Resuming from checkpoint: {len(disease_features)} diseases already processed")
        else:
            disease_features = {}
        
        disease_names = [d['name'] for d in diseases]
        remaining = [d for d in disease_names if d not in disease_features]
        
        if not remaining:
            print("All diseases already processed")
            return disease_features
        
        print("\n" + "="*80)
        print("STEP 2: LLM Feature Generation")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Generating features for {len(remaining)} diseases...")
        
        # Calculate time estimate based on rate limits
        if self.use_gemini:
            # Rate limits vary widely by model
            if 'lite' in self.model:
                # Flash-lite models: 4,000 RPM (extremely fast!)
                delay_seconds = 0.1
            elif '2.5-flash' in self.model or '2.0-flash' in self.model:
                # Flash models: 1,000-2,000 RPM (very fast)
                delay_seconds = 0.1
            elif '2.5-pro' in self.model:
                # Pro models: 150 RPM (fast)
                delay_seconds = 0.5
            elif '1.5-flash' in self.model or '1.5-pro' in self.model:
                # Legacy 1.5: 15 RPM
                delay_seconds = 4.5
            else:
                # Conservative default
                delay_seconds = 1.0
            
            estimated_time = len(remaining) * delay_seconds / 60
            print(f"(Using Gemini API: ~{estimated_time:.1f} minutes)\n")
        else:
            delay_seconds = 0.5
            print(f"(~{len(remaining) * 5} seconds estimated)\n")
        
        for disease_name in tqdm(remaining, desc="Generating features"):
            features = self._generate_single_disease(disease_name)
            
            if features:
                disease_features[disease_name] = features
                
                # Save checkpoint after each disease
                with open(checkpoint_file, 'w') as f:
                    json.dump(disease_features, f, indent=2)
            
            # Rate limiting delay
            time.sleep(delay_seconds)
        
        # Leverage the DDXPLus dataset to get the symptoms for each disease and add them to the disease_features
        # loop through all .json files in dataset/df/train 
        for disease_name, disease_feature in tqdm(disease_features.items(), desc="Scraping symptoms"):
            symptoms = self._scrape_symptoms(disease_name)
            disease_feature['layperson_symptoms'] = symptoms
        
        print(f"\nFeatures generated for {len(disease_features)} diseases")
        return disease_features
    
    def _get_clean_symptom_prompt(self, symptom_text):
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
        
    def _clean_ddxplus_symptom(self, symptom_text):
        """
        Parses DDXPlus formatted symptoms into clinical keywords.
        """
        prompt = self._get_clean_symptom_prompt(symptom_text)
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
        # The assistant message is always the last dict
        assistant_msg = gen[-1]["content"]
        # print(f'symptom text: {symptom_text}, assistant_msg: {assistant_msg}')
        # If this one message has multiple keywords separated by commas, parse them into a list
        if ',' in assistant_msg:
            return assistant_msg.split(',')
        else:
            return [assistant_msg]

    def _scrape_symptoms(self, disease_name):
        """
        Goes through each .json file and look at the 'All Features" key
        - Use the same small LLM to summarize each feature into few key words
        - Use the same symptoms deduplication logic to deduplicate the symptoms
        - return a list of critical symptoms for the disease
        
        """
        cleaned_symptoms = []
        for file in tqdm(self.ddxplus_files, desc=f"Scraping symptoms for {disease_name}"):
            with open(os.path.join(self.ddxplus_dir, file), 'r') as f:
                data = json.load(f)
            # if not the disease as Processed Diagnosis, skip
            if data['Processed Diagnosis'] != disease_name:
                continue
            all_features = data['All Features']
            for feature, value in all_features.items():
                cleaned_symptom = self._clean_ddxplus_symptom(feature)
                cleaned_symptoms.extend(cleaned_symptom)
        
        cleaned_symptoms = self._semantic_deduplication_symptoms(cleaned_symptoms)
        return cleaned_symptoms

    def _semantic_deduplication_symptoms(self, symptoms, similarity_threshold=0.75):
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
        # Cosine distance = 1 - Cosine Similarity
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

        # 4. Consolidate: Pick the shortest term from each cluster
        deduplicated = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            # Get all symptoms in this cluster
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_items = [symptoms[i] for i in cluster_indices]
            
            shortest_term = min(cluster_items, key=len)
            deduplicated.append(shortest_term)
            
        return deduplicated
    
    def _generate_single_disease(self, disease_name):
        """Generate features for a single disease using few-shot prompting"""
        
        system_prompt = """You are a medical JSON generator. Output ONLY valid JSON objects. Never explain, never add text."""

        user_prompt = f"""Generate medical features in JSON format.
        
For 'critical_symptoms', include both the medical term AND 1-2 common layperson descriptions.
Example:
"critical_symptoms": ["Pyrosis (Heartburn, Burning chest)", "Dyspnea (Shortness of breath, Can't breathe)"]

Example 1:
Disease: Pneumonia
{{
  "critical_symptoms": ["Fever", "Productive cough", "Chest pain", "Dyspnea", "Fatigue"],
  "diagnostic_features": ["Consolidation on chest X-ray", "Elevated WBC count", "Crackles on auscultation"],
  "distinguishing_features": ["Acute onset vs tuberculosis", "Fever and productive cough vs bronchitis", "Lobar consolidation pattern"],
  "risk_factors": ["Age >65 years", "Immunocompromised state", "Chronic lung disease"]
}}

Example 2:
Disease: Myocardial infarction
{{
  "critical_symptoms": ["Chest pain", "Dyspnea", "Diaphoresis", "Nausea", "Arm pain"],
  "diagnostic_features": ["Elevated troponin", "ST elevation on ECG", "Regional wall motion abnormality"],
  "distinguishing_features": ["Troponin elevation vs unstable angina", "ST changes vs pericarditis", "Acute onset severe pain"],
  "risk_factors": ["Hypertension", "Diabetes", "Smoking history", "Family history"]
}}

Now generate for this disease:
Disease: {disease_name}
{{
  "critical_symptoms": ["""

        try:
            if self.use_gemini:
                from authentication import google_api_key
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        parsed = generate_json_with_gemini(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            api_key=google_api_key,
                            model_name=self.hf_model,
                            temperature=0.3,
                            max_tokens=800
                        )
                        
                        if not parsed:
                            print(f"\nGemini returned None for {disease_name}")
                        
                        return parsed
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        if "429" in error_str or "quota" in error_str.lower():
                            if attempt < max_retries - 1:
                                wait_time = 30 * (attempt + 1)
                                print(f"\nRate limit hit for {disease_name}. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"\nRate limit exceeded after {max_retries} retries for {disease_name}")
                                return None
                        else:
                            print(f"\nGemini error for {disease_name}: {e}")
                            return None
                
                return None
            else:
                # Use local LLM
                response = generate_diagnosis_with_local_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=self.hf_model,
                    max_new_tokens=400
                )

                if not response.strip().startswith('{'):
                    response = '{\n  "critical_symptoms": [' + response
                
                if not response.strip().endswith('}'):
                    if ']' in response:
                        response = response + '\n}'
                
                parsed = self._parse_json_response(response)
                
                if not parsed:
                    print(f"\nFailed to parse response for {disease_name}")
                
                return parsed
                
        except Exception as e:
            print(f"\nError processing {disease_name}: {e}")
            return None
    
    def _parse_json_response(self, response):
        """Parse JSON from LLM response - handles various formats"""

        
        try:
            try:
                return json.loads(response.strip())
            except:
                pass
            
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                try:
                    return json.loads(json_str)
                except:
                    pass
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                try:
                    return json.loads(json_str)
                except:
                    pass
            
            json_start = response.find('{')
            
            if json_start != -1:
                brace_count = 0
                json_end = -1
                
                for i in range(json_start, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    json_str = response[json_start:json_end]
                    
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        json_str_clean = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        if e.pos and e.pos < len(json_str):
                            truncated = json_str[:e.pos]
                            last_comma = truncated.rfind(',')
                            if last_comma > 0:
                                json_str_partial = json_str[:last_comma] + '\n}'
                                try:
                                    result = json.loads(json_str_partial)
                                    print(f"Parsed partial taxonomy ({len(result)} diseases)")
                                    return result
                                except:
                                    pass
                        
                        try:
                            return json.loads(json_str_clean)
                        except:
                            pass

            if '[' in response and ']' in response:
                arr_start = response.find('[')
                bracket_count = 0
                arr_end = -1
                
                for i in range(arr_start, len(response)):
                    if response[i] == '[':
                        bracket_count += 1
                    elif response[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            arr_end = i + 1
                            break
                
                if arr_end != -1:
                    json_str = response[arr_start:arr_end]
                    try:
                        return json.loads(json_str)
                    except:
                        pass
            
            return None
            
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"\nWarning: Could not parse JSON: {e}")
            return None
    
    def build_kg(self, taxonomy, disease_features):
        """Step 3: Build knowledge graph from taxonomy and features"""
        
        print("\n" + "="*80)
        print("STEP 3: Building Knowledge Graph")
        print("="*80)
        
        kg_triples = []
        
        # Build hierarchical relationships
        for disease, info in taxonomy.items():
            level_1 = info.get('level_1', 'Unknown')
            level_2 = info.get('level_2', 'Unknown')
            level_3 = disease
            
            kg_triples.append({
                'subject': level_3,
                'relation': 'is_a',
                'object': level_2
            })
            kg_triples.append({
                'subject': level_2,
                'relation': 'is_a',
                'object': level_1
            })
            
            # Similarity relationships
            for similar in info.get('similar_diseases', []):
                kg_triples.append({
                    'subject': level_3,
                    'relation': 'similar_to',
                    'object': similar
                })
        
        # Build feature relationships
        for disease, features in disease_features.items():
            if not features:
                continue
            
            for symptom in features.get('critical_symptoms', []):
                kg_triples.append({
                    'subject': disease,
                    'relation': 'has_symptom',
                    'object': symptom
                })
            
            for feature in features.get('diagnostic_features', []):
                kg_triples.append({
                    'subject': disease,
                    'relation': 'has_feature',
                    'object': feature
                })
            
            for diff in features.get('distinguishing_features', []):
                kg_triples.append({
                    'subject': disease,
                    'relation': 'distinguished_by',
                    'object': diff
                })
            
            for risk in features.get('risk_factors', []):
                kg_triples.append({
                    'subject': disease,
                    'relation': 'risk_factor',
                    'object': risk
                })
            
            for symptom in features.get('layperson_symptoms', []):
                kg_triples.append({
                    'subject': disease,
                    'relation': 'has_symptom',
                    'object': symptom
                })
        
        print(f"Built {len(kg_triples)} knowledge graph triples")
        
        # Statistics
        relation_counts = {}
        for triple in kg_triples:
            rel = triple['relation']
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        print("\nRelation types:")
        for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
            print(f"  {relation:25s}: {count:4d}")
        
        return kg_triples
    
    def save_kg(self, kg_triples, taxonomy, disease_features):
        """Save knowledge graph in multiple formats"""
        
        # Save as Excel (for MedRAG compatibility)
        kg_df = pd.DataFrame(kg_triples)
        kg_excel = self.output_dir / 'knowledge_graph_DDXPlus_LLM_v2.xlsx'
        kg_df.to_excel(kg_excel, index=False)
        
        # Save as JSON (for inspection)
        kg_json = self.output_dir / 'knowledge_graph_DDXPlus_LLM_v2.json'
        with open(kg_json, 'w') as f:
            json.dump(kg_triples, f, indent=2)
        
        # Save complete package
        complete_package = {
            'taxonomy': taxonomy,
            'disease_features': disease_features,
            'kg_triples': kg_triples,
            'metadata': {
                'model': self.model,
                'num_diseases': len(taxonomy),
                'num_triples': len(kg_triples),
                'creation_method': 'LLM-generated (top-down)'
            }
        }
        
        package_file = self.output_dir / 'complete_kg_package_v2.json'
        with open(package_file, 'w') as f:
            json.dump(complete_package, f, indent=2)
        
        print("\n" + "="*80)
        print("Knowledge Graph Saved!")
        print("="*80)
        print(f"\nOutput files in {self.output_dir}:")
        print(f"  {kg_excel.name} - For MedRAG (Excel format)")
        print(f"  {kg_json.name} - For inspection (JSON)")
        print(f"  {package_file.name} - Complete package")
        print(f"  taxonomy_checkpoint.json - Disease categories")
        print(f"  features_checkpoint.json - Disease features")
        
        return kg_excel
    
    def run(self):
        """Run complete KG building pipeline"""
        
        print("="*80)
        print("LLM-Based Knowledge Graph Construction for DDXPlus")
        print("Top-Down Approach: LLM as Medical Textbook")
        print("="*80)
        print(f"\nModel: {self.model}")
        print(f"Output: {self.output_dir}\n")
        
        # Load diseases
        print("Loading DDXPlus diseases...")
        diseases = self.load_diseases()
        print(f"Loaded {len(diseases)} diseases\n")
        
        # Step 1: Clustering
        taxonomy = self.cluster_diseases(diseases, resume=True)
        if not taxonomy:
            print("Failed to generate taxonomy")
            return None
        
        print(f"Clustered {len(taxonomy)} diseases into categories\n")
        
        # Step 2: Feature generation
        disease_features = self.generate_disease_features(diseases, resume=True)
        print(f"Generated features for {len(disease_features)} diseases\n")
        
        # Step 3: Build KG
        kg_triples = self.build_kg(taxonomy, disease_features)
        
        # Step 4: Save
        kg_file = self.save_kg(kg_triples, taxonomy, disease_features)
        
        print("\n" + "="*80)
        print("LLM-based Knowledge Graph Complete!")
        
        return kg_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build DDXPlus KG using LLM')
    parser.add_argument('--model', default='gemini-2.5-flash',
                       help='Model: gemini-2.5-flash (1000 RPM, recommended), gemini-2.5-flash-lite (4000 RPM, fastest), '
                            'gemini-2.5-pro (150 RPM, best quality), local/llama-3.1-8b')
    parser.add_argument('--output-dir', default='./dataset/llm_kg_v2',
                       help='Output directory for KG files')
    parser.add_argument('--use-local', action='store_true',
                       help='Force use of local LLM instead of Gemini')
    
    args = parser.parse_args()
    
    # Determine if using Gemini
    use_gemini = args.model.startswith('gemini') and not args.use_local
    
    builder = LLMKGBuilder(model=args.model, output_dir=args.output_dir, use_gemini=use_gemini)
    builder.run()

