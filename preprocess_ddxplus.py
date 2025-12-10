#!/usr/bin/env python3
"""
Preprocessing script to convert DDXPlus dataset into MedRAG format.
This script converts the CSV files from DDXPlus into individual JSON patient files
and creates a ground truth CSV with categories.
"""

import pandas as pd
import json
import os
from pathlib import Path
import ast

def load_evidence_metadata(evidences_file):
    with open(evidences_file, 'r') as f:
        evidences_dict = json.load(f)
    
    if isinstance(evidences_dict, dict):
        evidences = list(evidences_dict.values())
    else:
        evidences = evidences_dict
    
    evidence_map = {}
    for evidence in evidences:
        if isinstance(evidence, dict):
            evidence_name = evidence.get('name', '')
            evidence_map[evidence_name] = {
                'question_en': evidence.get('question_en', ''),
                'is_antecedent': evidence.get('is_antecedent', False),
                'data_type': evidence.get('data_type', ''),
                'value_meaning': evidence.get('value_meaning', {})
            }
    return evidence_map

def load_conditions_metadata(conditions_file):
    with open(conditions_file, 'r') as f:
        conditions_dict = json.load(f)
    
    if isinstance(conditions_dict, dict):
        conditions = list(conditions_dict.values())
    else:
        conditions = conditions_dict
    
    condition_map = {}
    for condition in conditions:
        if isinstance(condition, dict):
            cond_name = condition.get('condition_name', '')
            condition_map[cond_name] = {
                'icd10': condition.get('icd10-id', ''),
                'severity': condition.get('severity', ''),
                'cond_name_eng': condition.get('cond-name-eng', cond_name)
            }
    return condition_map

def parse_evidences(evidences_str, evidence_map):
    try:
        evidences = ast.literal_eval(evidences_str)
    except:
        return {}
    
    parsed = {}
    symptoms = []
    antecedents = []
    
    for evidence in evidences:
        if '_@_' in evidence:
            parts = evidence.split('_@_')
            evidence_code = parts[0]
            value = parts[1]
            
            if evidence_code in evidence_map:
                metadata = evidence_map[evidence_code]
                question = metadata['question_en']
                
                value_meaning = metadata.get('value_meaning', {}).get(value, {})
                value_text = value_meaning.get('en', value)
                
                if metadata['is_antecedent']:
                    antecedents.append(f"{question}: {value_text}")
                else:
                    symptoms.append(f"{question}: {value_text}")
                
                parsed[question] = value_text
        else:
            if evidence in evidence_map:
                metadata = evidence_map[evidence]
                question = metadata['question_en']
                
                if metadata['is_antecedent']:
                    antecedents.append(question)
                else:
                    symptoms.append(question)
                
                parsed[question] = "Yes"
    
    return {
        'all_features': parsed,
        'symptoms': symptoms,
        'antecedents': antecedents
    }

def create_patient_json(row, patient_id, evidence_map, output_dir):
    evidences_parsed = parse_evidences(row['EVIDENCES'], evidence_map)
    
    try:
        diff_diagnosis = ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS'])
    except:
        diff_diagnosis = []
    
    patient_data = {
        "Participant No.": patient_id,
        "AGE": int(row['AGE']),
        "SEX": row['SEX'],
        "PATHOLOGY": row['PATHOLOGY'],
        "Processed Diagnosis": row['PATHOLOGY'],
        "Symptoms": evidences_parsed['symptoms'],
        "Antecedents": evidences_parsed['antecedents'],
        "All Features": evidences_parsed['all_features'],
        "DIFFERENTIAL_DIAGNOSIS": diff_diagnosis,
        "INITIAL_EVIDENCE": row.get('INITIAL_EVIDENCE', ''),
        "Raw_EVIDENCES": row['EVIDENCES']
    }
    
    output_file = os.path.join(output_dir, f"participant_{patient_id}.json")
    with open(output_file, 'w') as f:
        json.dump(patient_data, f, indent=2)
    
    return patient_data

def create_ground_truth_csv(all_patients, output_file, condition_map):
    """Create a ground truth CSV file with categories"""
    
    # Define hierarchical categories for DDXPlus diseases
    diagnosis_to_category = {
        # Respiratory
        "Bronchitis": ("Respiratory Diseases", "Lower Respiratory", "Bronchitis"),
        "Pneumonia": ("Respiratory Diseases", "Lower Respiratory", "Pneumonia"),
        "URTI": ("Respiratory Diseases", "Upper Respiratory", "URTI"),
        "Bronchiectasis": ("Respiratory Diseases", "Lower Respiratory", "Bronchiectasis"),
        "Tuberculosis": ("Infectious Diseases", "Bacterial Infections", "Tuberculosis"),
        "Influenza": ("Infectious Diseases", "Viral Infections", "Influenza"),
        "Viral pharyngitis": ("Respiratory Diseases", "Upper Respiratory", "Viral pharyngitis"),
        "Acute laryngitis": ("Respiratory Diseases", "Upper Respiratory", "Acute laryngitis"),
        "Whooping cough": ("Respiratory Diseases", "Lower Respiratory", "Whooping cough"),
        "Bronchiolitis": ("Respiratory Diseases", "Lower Respiratory", "Bronchiolitis"),
        
        # Cardiovascular
        "Myocarditis": ("Cardiovascular Diseases", "Heart Diseases", "Myocarditis"),
        "Pericarditis": ("Cardiovascular Diseases", "Heart Diseases", "Pericarditis"),
        "Unstable angina": ("Cardiovascular Diseases", "Coronary Artery Disease", "Unstable angina"),
        "Stable angina": ("Cardiovascular Diseases", "Coronary Artery Disease", "Stable angina"),
        "Possible NSTEMI / STEMI": ("Cardiovascular Diseases", "Coronary Artery Disease", "Possible NSTEMI / STEMI"),
        "PSVT": ("Cardiovascular Diseases", "Arrhythmias", "PSVT"),
        "Atrial fibrillation": ("Cardiovascular Diseases", "Arrhythmias", "Atrial fibrillation"),
        "Pulmonary embolism": ("Cardiovascular Diseases", "Vascular Diseases", "Pulmonary embolism"),
        
        # Gastrointestinal
        "GERD": ("Gastrointestinal Diseases", "Esophageal Disorders", "GERD"),
        "Pancreatic neoplasm": ("Gastrointestinal Diseases", "Neoplasms", "Pancreatic neoplasm"),
        "Scombroid food poisoning": ("Infectious Diseases", "Food Poisoning", "Scombroid food poisoning"),
        "Boerhaave": ("Gastrointestinal Diseases", "Esophageal Disorders", "Boerhaave"),
        "Inguinal hernia": ("Gastrointestinal Diseases", "Hernias", "Inguinal hernia"),
        
        # Neurological
        "Cluster headache": ("Neurological Diseases", "Headache Disorders", "Cluster headache"),
        "Myasthenia gravis": ("Neurological Diseases", "Neuromuscular Disorders", "Myasthenia gravis"),
        "Guillain-Barré syndrome": ("Neurological Diseases", "Neuromuscular Disorders", "Guillain-Barré syndrome"),
        "Acute dystonic reactions": ("Neurological Diseases", "Movement Disorders", "Acute dystonic reactions"),
        
        # Infectious
        "HIV (initial infection)": ("Infectious Diseases", "Viral Infections", "HIV (initial infection)"),
        "Chagas": ("Infectious Diseases", "Parasitic Infections", "Chagas"),
        "Ebola": ("Infectious Diseases", "Viral Infections", "Ebola"),
        "Acute otitis media": ("Infectious Diseases", "Ear Infections", "Acute otitis media"),
        "Chronic rhinosinusitis": ("Respiratory Diseases", "Sinus Disorders", "Chronic rhinosinusitis"),
        "Acute rhinosinusitis": ("Respiratory Diseases", "Sinus Disorders", "Acute rhinosinusitis"),
        "Allergic sinusitis": ("Respiratory Diseases", "Sinus Disorders", "Allergic sinusitis"),
        
        # Autoimmune
        "SLE": ("Autoimmune Diseases", "Systemic Autoimmune", "SLE"),
        "Sarcoidosis": ("Autoimmune Diseases", "Granulomatous Diseases", "Sarcoidosis"),
        "Anaphylaxis": ("Allergic Disorders", "Severe Allergic Reactions", "Anaphylaxis"),
        
        # Other
        "Anemia": ("Hematological Diseases", "Blood Disorders", "Anemia"),
        "Panic attack": ("Psychiatric Diseases", "Anxiety Disorders", "Panic attack"),
        "Spontaneous rib fracture": ("Musculoskeletal Diseases", "Bone Disorders", "Spontaneous rib fracture"),
        "Spontaneous pneumothorax": ("Respiratory Diseases", "Pleural Disorders", "Spontaneous pneumothorax"),
    }
    
    # Create rows for CSV
    rows = []
    for patient in all_patients:
        pathology = patient['PATHOLOGY']
        category_info = diagnosis_to_category.get(pathology, ("Unknown", "Unknown", pathology))
        
        # Get differential diagnosis as a list of disease names
        diff_diag_list = [item[0] for item in patient.get('DIFFERENTIAL_DIAGNOSIS', [])]
        
        row = {
            'PATIENT_ID': patient['Participant No.'],
            'Participant No.': patient['Participant No.'],
            'AGE': patient['AGE'],
            'SEX': patient['SEX'],
            'Symptoms': '; '.join(patient.get('Symptoms', [])),
            'Antecedents': '; '.join(patient.get('Antecedents', [])),
            'Processed Diagnosis': pathology,
            'Level 2': category_info[1],
            'Level 1': category_info[0],
            'Filtered_Diagnoses': str(diff_diag_list),
            'DIFFERENTIAL_DIAGNOSIS': str(patient.get('DIFFERENTIAL_DIAGNOSIS', []))
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Ground truth CSV created with {len(df)} patients: {output_file}")
    return df

def main(max_samples=None):
    """Main preprocessing function
    
    Args:
        max_samples: Maximum number of samples to process per split (None = all)
    """
    
    # Paths
    ddxplus_dir = Path('./DDXPlus')
    dataset_dir = Path('./dataset')
    
    # Create output directories
    train_dir = dataset_dir / 'df' / 'train'
    val_dir = dataset_dir / 'df' / 'val'
    test_dir = dataset_dir / 'df' / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("Loading evidence and condition metadata...")
    evidence_map = load_evidence_metadata(ddxplus_dir / 'release_evidences.json')
    condition_map = load_conditions_metadata(ddxplus_dir / 'release_conditions.json')
    
    # Process each split
    splits = {
        'train': (ddxplus_dir / 'release_train_patients.csv', train_dir),
        'val': (ddxplus_dir / 'release_validate_patients.csv', val_dir),
        'test': (ddxplus_dir / 'release_test_patients.csv', test_dir)
    }
    
    all_data = {'train': [], 'val': [], 'test': []}
    
    for split_name, (csv_file, output_dir) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping {split_name} split")
            continue
        
        df = pd.read_csv(csv_file)
        
        # Limit samples if specified
        if max_samples is not None:
            df = df.head(max_samples)
            print(f"Limited to {max_samples} samples (total available: {len(pd.read_csv(csv_file))})")
        
        print(f"Processing {len(df)} patients in {split_name} split...")
        
        # Create JSON files for each patient
        for idx, row in df.iterrows():
            patient_id = idx + 1  # Start from 1
            patient_data = create_patient_json(row, patient_id, evidence_map, output_dir)
            all_data[split_name].append(patient_data)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} patients...")
        
        print(f"Created {len(all_data[split_name])} JSON files in {output_dir}")
    
    # Create ground truth CSV files for each split
    print("\nCreating ground truth CSV files...")
    for split_name in ['train', 'val', 'test']:
        if all_data[split_name]:
            output_csv = dataset_dir / f'AI_Data_Set_with_Categories_{split_name}.csv'
            create_ground_truth_csv(all_data[split_name], output_csv, condition_map)
    
    # Also create a combined ground truth file if needed
    all_combined = all_data['train'] + all_data['val'] + all_data['test']
    if all_combined:
        combined_csv = dataset_dir / 'AI_Data_Set_with_Categories_all.csv'
        create_ground_truth_csv(all_combined, combined_csv, condition_map)
    
    print("\nPreprocessing complete!")
    print(f"  Train: {len(all_data['train'])} patients")
    print(f"  Val: {len(all_data['val'])} patients")
    print(f"  Test: {len(all_data['test'])} patients")
    print(f"\nOutput directories:")
    print(f"  Train JSON files: {train_dir}")
    print(f"  Val JSON files: {val_dir}")
    print(f"  Test JSON files: {test_dir}")
    print(f"  Ground truth CSVs: {dataset_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess DDXPlus dataset for MedRAG')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process per split (default: all)')
    
    args = parser.parse_args()
    
    if args.max_samples:
        print(f"Processing with max_samples={args.max_samples} per split")
    else:
        print("Processing ALL samples (this may take a while...)")
    
    main(max_samples=args.max_samples)

