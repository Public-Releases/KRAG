#!/usr/bin/env python3
"""
Script to run MedRAG on DDXPlus dataset (train, val, or test splits)
Usage:
    python run_ddxplus.py --split train --topk 1 --topn 1 --matchn 5 --max_samples 100
    python run_ddxplus.py --split test --topk 1 --topn 1 --matchn 5
"""

import os
import re
import json
import argparse
import pandas as pd
import traceback
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from main_MedRAG import (
    get_query_embedding, 
    Faiss, 
    extract_diagnosis, 
    generate_diagnosis_report, 
    save_results_to_csv, 
    get_additional_info_from_level_2,
    KG_preprocess, 
    get_embeddings
)

def run_medrag_on_split(split='test', topk=1, top_n=1, match_n=5, max_samples=None, model='local/llama-3.1-8b', kg_path='./dataset/llm_kg/knowledge_graph_DDXPlus_LLM.xlsx'):
    """
    Run MedRAG on a specific data split
    
    Args:
        split: 'train', 'val', or 'test'
        topk: Number of similar patients to retrieve
        top_n: Top N categories to consider
        match_n: Number of matches for KG traversal
        max_samples: Maximum number of samples to process (None for all)
    """
    
    # Paths
    dataset_dir = Path('./dataset')
    train_folder_path = dataset_dir / 'df' / 'train'
    test_folder_path = dataset_dir / 'df' / split
    ground_truth_file_path = dataset_dir / f'AI_Data_Set_with_Categories_{split}.csv'
    
    augmented_features_path = Path(kg_path)
    if not test_folder_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {test_folder_path}\n"
            f"Please run 'python preprocess_ddxplus.py' first to prepare the data."
        )
    
    if not ground_truth_file_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_file_path}\n"
            f"Please run 'python preprocess_ddxplus.py' first to prepare the data."
        )
     
    if not augmented_features_path.exists():
        print(f"Warning: Knowledge graph not found at {augmented_features_path}")
        exit()
    
    print(f"\n{'='*80}")
    print(f"Running MedRAG on DDXPlus {split.upper()} split")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Split: {split}")
    print(f"  - Model: {model}")
    print(f"  - Top-K retrieval: {topk}")
    print(f"  - Top-N categories: {top_n}")
    print(f"  - Match-N for KG: {match_n}")
    print(f"  - Max samples: {max_samples if max_samples else 'All'}")
    print(f"\nPaths:")
    print(f"  - Training data (for retrieval): {train_folder_path}")
    print(f"  - Test data: {test_folder_path}")
    print(f"  - Ground truth: {ground_truth_file_path}")
    print(f"  - Knowledge graph: {augmented_features_path}")
    print(f"{'='*80}\n")
    
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_file_path, header=0)
    
    # Get training documents for retrieval
    print("Loading training documents for retrieval...")
    documents = [os.path.join(train_folder_path, file_name) 
                for file_name in os.listdir(train_folder_path) 
                if os.path.isfile(os.path.join(train_folder_path, file_name))]
    print(f"Loaded {len(documents)} training documents")
    
    # Get or create document embeddings
    document_embeddings_file_path = dataset_dir / 'document_embeddings_ddxplus.npy'
    
    if document_embeddings_file_path.exists():
        print(f"Loading pre-computed document embeddings from {document_embeddings_file_path}")
        import numpy as np
        document_embeddings = np.load(document_embeddings_file_path)
    else:
        print("Computing document embeddings (this may take a while)...")
        document_embeddings = get_embeddings(documents)
        import numpy as np
        np.save(document_embeddings_file_path, document_embeddings)
        print(f"Saved document embeddings to {document_embeddings_file_path}")
    
    # Get test files
    test_files = sorted([f for f in os.listdir(test_folder_path) if f.endswith('.json')])
    
    if max_samples:
        test_files = test_files[:max_samples]
    
    print(f"\nProcessing {len(test_files)} test cases...")
    
    # Results storage
    results = []
    
    # Process each test case
    for file_name in tqdm(test_files, desc="Processing patients"):
        file_path = os.path.join(test_folder_path, file_name)
        
        # Extract participant number from filename
        participant_no = int(file_name.replace('participant_', '').replace('.json', ''))
        
        # Load patient case
        with open(file_path, 'r') as file:
            new_patient_case = json.load(file)
        
        # Remove ground truth fields to prevent data leakage
        from main_MedRAG import remove_ground_truth_fields
        cleaned_patient_case = remove_ground_truth_fields(new_patient_case)
        # Convert to query string
        query = json.dumps(cleaned_patient_case)
        
        success = False
        retry_count = 0
        max_retries = 3
        
        while not success and retry_count < max_retries:
            try:
                # 1. Get embedding of the given EHR
                query_embedding = get_query_embedding(query)
                
                # 2. Search EHR database for most similar past cases
                indices = Faiss(document_embeddings, query_embedding, k=topk)
                retrieved_documents = [documents[i] for i in indices[0]]
                
                # 3. Extract relevant information from retrieved documents
                final_retrieved_info = []
                for retrieved_document in retrieved_documents:
                    with open(retrieved_document, 'r') as file:
                        patient_case = json.load(file)
                        # For DDXPlus, we want different fields than the chronic pain dataset
                        filtered_patient_case_dict = {
                            key: patient_case[key] for key in [
                                "Processed Diagnosis",
                                "Symptoms",
                                "Antecedents",
                                "AGE",
                                "SEX"
                            ] if key in patient_case
                        }
                        final_retrieved_info.append(filtered_patient_case_dict)
                
                # 4. Get true diagnosis from ground truth
                true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == participant_no]
                
                if true_diagnosis_row.empty:
                    print(f"\nWarning: True diagnosis for patient_{participant_no} not found in ground truth")
                    results.append([participant_no, '', '', '', ''])
                    success = True
                    continue
                
                true_diagnosis = true_diagnosis_row['Processed Diagnosis'].values[0]
                true_level2 = true_diagnosis_row['Level 2'].values[0] if 'Level 2' in true_diagnosis_row.columns else ''
                true_level1 = true_diagnosis_row['Level 1'].values[0] if 'Level 1' in true_diagnosis_row.columns else ''
                
                # 5. Generate diagnosis report using KG-elicited reasoning
                generated_report, kg_info, clean_symptoms, winning_category = generate_diagnosis_report(
                    augmented_features_path,
                    query, 
                    final_retrieved_info, 
                    participant_no,
                    top_n=top_n,
                    match_n=match_n,
                    model=model
                )
                
                # 6. Extract diagnosis from report
                generated_diagnosis = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report)
                
                if not generated_diagnosis:
                    print(f"\nWarning: Could not extract diagnosis for patient_{participant_no}")
                    results.append([participant_no, '', true_diagnosis, true_level2, true_level1, generated_report])
                else:
                    results.append([
                        participant_no, 
                        generated_diagnosis[0], 
                        true_diagnosis, 
                        true_level2,
                        true_level1,
                        generated_report,
                        kg_info,
                        clean_symptoms,
                        winning_category,
                        final_retrieved_info
                    ])
                
                success = True
                
            except Exception as e:
                retry_count += 1
                print(f"\nError processing patient_{participant_no} (attempt {retry_count}/{max_retries}): {e}")
                print(f"Error type: {type(e).__name__}")
                print("Traceback (most recent call last):")
                traceback.print_exc()
                if retry_count >= max_retries:
                    print(f"Failed to process patient_{participant_no} after {max_retries} attempts")
                    results.append([participant_no, '', true_diagnosis if 'true_diagnosis' in locals() else '', '', '', f"Error: {str(e)}"])
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"results/{split}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results with descriptive filename
    model_name_clean = model.replace('/', '_')
    output_filename = f"results_{split}_{model_name_clean}_topk{topk}_topn{top_n}_matchn{match_n}.csv"
    output_file = os.path.join(results_dir, output_filename)
    
    df = pd.DataFrame(results, columns=[
        'Participant No.', 
        'Generated Diagnosis', 
        'True Diagnosis', 
        'True Level 2',
        'True Level 1',
        'Generated Report',
        'KG Info',
        'Clean Symptoms',
        'Winning Category',
        'Final Retrieved Info'
    ])
    df.to_csv(output_file, index=False)
    
    # Also save experiment configuration
    config = {
        'timestamp': timestamp,
        'split': split,
        'model': model,
        'topk': topk,
        'top_n': top_n,
        'match_n': match_n,
        'max_samples': max_samples,
        'total_processed': len(results),
        'successful': len([r for r in results if r[1]]),
        'output_file': output_file
    }
    
    config_file = os.path.join(results_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_dir}")
    print(f"  - Results CSV: {output_filename}")
    print(f"  - Config: config.json")
    print(f"Total cases processed: {len(results)}")
    print(f"Successful diagnoses: {len([r for r in results if r[1]])}")
    print(f"{'='*80}\n")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Run MedRAG on DDXPlus dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which data split to use (default: test)')
    parser.add_argument('--model', type=str, default='local/llama-3.1-8b',
                       help='LLM model to use (default: local/llama-3.1-8b). '
                            'Options: local/llama-3.1-8b, local/openbiollm-8b (medical-specialized), '
                            'local/llama-3.1-70b, local/llama-3.2-3b, local/mistral-7b, '
                            'gpt-4o, gpt-4o-mini, gpt-3.5-turbo-0125')
    parser.add_argument('--topk', type=int, default=1,
                       help='Number of similar patients to retrieve (default: 1)')
    parser.add_argument('--topn', type=int, default=1,
                       help='Top N categories to consider (default: 1)')
    parser.add_argument('--matchn', type=int, default=5,
                       help='Number of matches for KG traversal (default: 5)')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--kg-path', type=str, default='./dataset/llm_kg/knowledge_graph_DDXPlus_LLM.xlsx',
                       help='Path to the knowledge graph (default: ./dataset/llm_kg/knowledge_graph_DDXPlus_LLM.xlsx)')
    args = parser.parse_args()
    
    run_medrag_on_split(
        split=args.split,
        topk=args.topk,
        top_n=args.topn,
        match_n=args.matchn,
        max_samples=args.max_samples,
        model=args.model,
        kg_path=args.kg_path
    )

if __name__ == "__main__":
    main()

