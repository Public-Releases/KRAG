import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from main_MedRAG import get_query_embedding, Faiss, get_embeddings, get_system_prompt_for_RAGKG
from authentication import ob_path, test_folder_path, ground_truth_file_path

def generate_diagnosis_baseline(query, model='local/llama-3.1-8b'):
    """Generate diagnosis without RAG or KG - baseline"""
    
    system_prompt = get_system_prompt_for_RAGKG()
    
    user_prompt = f"Patient information:\n{query}\n\nProvide your diagnosis in the specified format."
    
    if model.startswith('local/'):
        from local_llm_inference import generate_diagnosis_with_local_llm
        
        model_map = {
            'local/llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'local/llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
            'local/openbiollm-8b': 'aaditya/Llama3-OpenBioLLM-8B',
        }
        hf_model = model_map.get(model, 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        
        response = generate_diagnosis_with_local_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=hf_model,
            max_new_tokens=500
        )
    else:
        from authentication import api_key
        import openai
        client = openai.OpenAI(api_key=api_key)
        response_obj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        response = response_obj.choices[0].message.content
    
    return response

def generate_diagnosis_rag_only(query, retrieved_docs, model='local/llama-3.1-8b'):
    """Generate diagnosis with RAG but without KG"""
    
    system_prompt = get_system_prompt_for_RAGKG()
    
    user_prompt = f"""Patient information:
{query}

Similar patient cases:
{retrieved_docs}

Provide your diagnosis in the specified format."""
    
    if model.startswith('local/'):
        from local_llm_inference import generate_diagnosis_with_local_llm
        model_map = {
            'local/llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'local/llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
            'local/openbiollm-8b': 'aaditya/Llama3-OpenBioLLM-8B',
        }
        hf_model = model_map.get(model, 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        
        response = generate_diagnosis_with_local_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=hf_model,
            max_new_tokens=500
        )
    else:
        from authentication import api_key
        import openai
        client = openai.OpenAI(api_key=api_key)
        response_obj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        response = response_obj.choices[0].message.content
    
    return response

def generate_diagnosis_kg_only(query, kg_context, model='local/llama-3.1-8b'):
    """Generate diagnosis with KG but without RAG"""
    
    system_prompt = get_system_prompt_for_RAGKG()
    
    user_prompt = f"""Patient information:
{query}

Medical Knowledge Graph Context:
{kg_context}

Provide your diagnosis in the specified format."""
    
    if model.startswith('local/'):
        from local_llm_inference import generate_diagnosis_with_local_llm
        model_map = {
            'local/llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'local/llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
            'local/openbiollm-8b': 'aaditya/Llama3-OpenBioLLM-8B',
        }
        hf_model = model_map.get(model, 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        
        response = generate_diagnosis_with_local_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=hf_model,
            max_new_tokens=500
        )
    else:
        from authentication import api_key
        import openai
        client = openai.OpenAI(api_key=api_key)
        response_obj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        response = response_obj.choices[0].message.content
    
    return response

def run_ablation_study(split='test', mode='baseline', topk=1, max_samples=None, model='local/llama-3.1-8b', augmented_features_path='./dataset/medrag_kg_ddxplus.xlsx'):
    """
    Run ablation study with different configurations
    """
    
    print(f"\n{'='*80}")
    print(f"Running Ablation Study: {mode.upper()}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Split: {split}")
    print(f"  - Mode: {mode}")
    print(f"  - Model: {model}")
    print(f"  - KG Path: {augmented_features_path}")
    if mode in ['full', 'rag_only']:
        print(f"  - Top-K retrieval: {topk}")
    print(f"  - Max samples: {max_samples or 'all'}")
    print(f"{'='*80}\n")
    
    # Load test data
    if split == 'test':
        test_data_path = test_folder_path
        ground_truth_file = './dataset/AI_Data_Set_with_Categories_test.csv'
    elif split == 'val':
        test_data_path = './dataset/df/val'
        ground_truth_file = './dataset/AI_Data_Set_with_Categories_val.csv'
    elif split == 's_to_d':
        test_data_path = './dataset/symptom_to_diagnosis'
    else:
        test_data_path = ob_path
        ground_truth_file = './dataset/AI_Data_Set_with_Categories_train.csv'
    
    # Load ground truth
    try:
        ground_truth = pd.read_csv(ground_truth_file)
    except:
        ground_truth = pd.read_csv(ground_truth_file_path, encoding='ISO-8859-1')
    
    # Get test files
    test_files = sorted([f for f in os.listdir(test_data_path) if f.endswith('.json')])
    if max_samples:
        test_files = test_files[:max_samples]
    
    print(f"Processing {len(test_files)} test cases...")
    
    # Load document embeddings and documents if using RAG
    if mode in ['full', 'rag_only']:
        print("Loading training documents for retrieval...")
        documents = [os.path.join(ob_path, f) for f in os.listdir(ob_path) if f.endswith('.json')]
        print(f"Loaded {len(documents)} training documents")
        
        # Load or generate embeddings
        embedding_file = './dataset/document_embeddings_ddxplus.npy'
        if os.path.exists(embedding_file):
            print("Loading pre-computed document embeddings...")
            document_embeddings = get_embeddings([open(doc).read() for doc in documents[:10]])  # Dummy call
            import numpy as np
            document_embeddings = np.load(embedding_file)
        else:
            print("Generating document embeddings (this may take a while)...")
            doc_texts = []
            for doc in tqdm(documents, desc="Reading documents"):
                with open(doc, 'r') as f:
                    doc_texts.append(json.dumps(json.load(f)))
            document_embeddings = get_embeddings(doc_texts)
            import numpy as np
            np.save(embedding_file, document_embeddings)
    
    # Load KG if using KG
    if mode in ['full', 'kg_only']:
        print("Loading Knowledge Graph...")
        from KG_Retrieve_DDXPlus import DDXPlusKGRetriever, get_additional_info_ddxplus
        kg_retriever = DDXPlusKGRetriever(kg_path=augmented_features_path)
    
    # Process test cases
    results = []
    
    # Import the utility function to remove ground truth fields
    from main_MedRAG import remove_ground_truth_fields
    
    for file_name in tqdm(test_files, desc="Processing patients"):
        file_path = os.path.join(test_data_path, file_name)
        participant_no = int(file_name.replace('participant_', '').replace('.json', ''))
        
        # Load patient case
        with open(file_path, 'r') as file:
            patient_case = json.load(file)
        
        # Remove ground truth fields before creating query
        cleaned_patient_case = remove_ground_truth_fields(patient_case)
        query = json.dumps(cleaned_patient_case)
        
        # placeholder values for report columns
        kg_context = ""
        clean_symptoms = []
        

        try:
            for i in range(3):
                if mode == 'baseline':
                    generated_report = generate_diagnosis_baseline(query, model)
                
                elif mode == 'rag_only':
                    query_embedding = get_query_embedding(query)
                    indices = Faiss(document_embeddings, query_embedding, k=topk)
                    retrieved_documents = [documents[i] for i in indices[0]]
                    
                    retrieved_info = []
                    for doc_path in retrieved_documents:
                        with open(doc_path, 'r') as f:
                            doc_data = json.load(f)
                            retrieved_info.append({
                                "Processed Diagnosis": doc_data.get('Processed Diagnosis', ''),
                                "Symptoms": doc_data.get('Symptoms', [])[:5],  # First 5 symptoms
                            })
                    
                    generated_report = generate_diagnosis_rag_only(query, str(retrieved_info), model)
                
                elif mode == 'kg_only':
                    kg_context, clean_symptoms, winning_category = get_additional_info_ddxplus(kg_retriever, query, top_n=3)
                    generated_report = generate_diagnosis_kg_only(query, kg_context, model)
                
                else:
                    from main_MedRAG import generate_diagnosis_report
                    
                    query_embedding = get_query_embedding(query)
                    indices = Faiss(document_embeddings, query_embedding, k=topk)
                    retrieved_documents = [documents[i] for i in indices[0]]
                    
                    retrieved_info = []
                    for doc_path in retrieved_documents:
                        with open(doc_path, 'r') as f:
                            doc_data = json.load(f)
                            retrieved_info.append({
                                key: doc_data[key] for key in [
                                    "Processed Diagnosis", "Symptoms", "Antecedents", "AGE", "SEX"
                                ] if key in doc_data
                            })
                    
                    generated_report, kg_context, clean_symptoms, winning_category = generate_diagnosis_report(
                        augmented_features_path, query, retrieved_info, participant_no,
                        top_n=1, match_n=5, model=model
                    )
                
                # Extract diagnosis
                diagnosis_match = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report)
                # print(f'Diagnosis match: {diagnosis_match}')
                generated_diagnosis = diagnosis_match[0] if diagnosis_match else ''
                if generated_diagnosis:
                    break
                else:
                    print(f'Generated diagnosis is empty. Retrying...')
                    continue
            
            # Get ground truth
            if split == 's_to_d':
                true_diagnosis = patient_case['Processed Diagnosis']
                true_level2 = ''
                true_level1 = ''
            else:
                true_row = ground_truth.loc[ground_truth['Participant No.'] == participant_no]
                if not true_row.empty:
                    true_diagnosis = true_row['Processed Diagnosis'].values[0]
                    true_level2 = true_row['Level 2'].values[0] if 'Level 2' in true_row.columns else ''
                    true_level1 = true_row['Level 1'].values[0] if 'Level 1' in true_row.columns else ''
                else:
                    print(f'Could not find true diagnosis for patient_{participant_no}')
                    true_diagnosis = ''
                    true_level2 = ''
                    true_level1 = ''
            
            results.append([
                participant_no,
                generated_diagnosis,
                true_diagnosis,
                true_level2,
                true_level1,
                generated_report,
                kg_context,
                query,
                clean_symptoms,
                winning_category
            ])
        
        except Exception as e:
            print(f"\nError processing patient_{participant_no}: {e}")
            results.append([participant_no, '', '', '', '', '', '', '', '', ''])
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path(f'results/{split}/{mode}/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df = pd.DataFrame(results, columns=[
        'Participant No.', 'Generated Diagnosis', 'True Diagnosis',
        'True Level 2', 'True Level 1', 'Generated Report', 'KG Context', 'Patient Query', 'Clean Symptoms', 'Winning Category'
    ])
    
    output_file = output_dir / f'results_{split}_{model.replace("/", "-")}_{mode}_topk{topk}.csv'
    df.to_csv(output_file, index=False)
    
    # Save config
    config = {
        'split': split,
        'mode': mode,
        'model': model,
        'topk': topk if mode in ['full', 'rag_only'] else 'N/A',
        'max_samples': max_samples,
        'timestamp': timestamp,
        'kg_path': str(augmented_features_path) if mode in ['full', 'kg_only'] else 'N/A'
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    ['lower chest pain', 'no travel outside country', 'hiatal hernia', 'pregnancy suspected', 'alcohol use', 'pain location unspecified', 'overweight', 'black stools', 'cough', 'worse when lying down alleviated while sitting up', 'burning sensation in throat, bitter taste in mouth', 'intensity 6', 'pain related to reason for consulting', 'pain appeared fast', 'hypochondrium pain', 'burning pain', 'tugging pain', 'haunting pain', 'pain sensitivity']
    success = df['Generated Diagnosis'].notna().sum()
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Results CSV: {output_file.name}")
    print(f"  - Config: config.json")
    print(f"Total cases processed: {len(results)}")
    print(f"Successful diagnoses: {success}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation study on MedRAG')
    parser.add_argument('--split', choices=['train', 'val', 'test', 's_to_d'], default='test')
    parser.add_argument('--mode', choices=['full', 'rag_only', 'kg_only', 'baseline'], 
                       default='baseline', help='Ablation mode')
    parser.add_argument('--model', default='local/llama-3.1-8b')
    parser.add_argument('--topk', type=int, default=1, help='Number of documents to retrieve')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--kg-path', default='./dataset/llm_kg_v2/knowledge_graph_DDXPlus_LLM_v2.xlsx')
    
    args = parser.parse_args()
    
    run_ablation_study(
        split=args.split,
        mode=args.mode,
        topk=args.topk,
        max_samples=args.max_samples,
        model=args.model,
        augmented_features_path=args.kg_path
    )

