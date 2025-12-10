import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import json

def clean_diagnosis_name(name):
    """Normalize diagnosis names for comparison"""
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Remove parenthetical explanations: "GERD (Gastroesophageal...)" -> "GERD"
    if '(' in name:
        name = name.split('(')[0].strip()
    
    # Normalize
    name = name.replace('_', ' ').replace('.', '').replace(' /', '/').replace('é', 'e').replace("-", ' ').strip().lower()
    
    return name

def flexible_match(row):
    """Flexible matching that handles variations, aliases, and substrings"""
    gen = str(row.get('Generated Diagnosis', '')).lower().strip()
    tru = str(row.get('True Diagnosis', '')).lower().strip()
    
    if not gen or not tru or gen == 'nan' or tru == 'nan':
        return False
    
    # Direct match
    if gen == tru:
        return True
    
    # Substring match (either direction)
    if tru in gen or gen in tru:
        return True
    
    # Aliases for common variations
    aliases = {
        "hiv initial infection": "hiv (initial infection)",
        "hiv initial": "hiv (initial infection)",
        "acute nasopharyngitis": "urti",
        "upper respiratory tract infection": "urti",
        "acute bronchitis": "bronchitis",
        "acute asthma exacerbation": "bronchospasm / acute asthma exacerbation",
        "bronchospasm": "bronchospasm / acute asthma exacerbation",
        "asthma exacerbation": "bronchospasm / acute asthma exacerbation",
        "gerd": "gerd (gastroesophageal reflux disease)",
        "gastroesophageal reflux": "gerd (gastroesophageal reflux disease)",
        "gastroesophageal reflux disease": "gerd (gastroesophageal reflux disease)",
    }
    
    # Check aliases (both directions)
    if aliases.get(gen) == tru or aliases.get(tru) == gen:
        return True
    
    # Check if alias matches either side
    for alias_key, alias_val in aliases.items():
        if (gen == alias_key and tru == alias_val) or (gen == alias_val and tru == alias_key):
            return True
    
    return False

def calculate_accuracy(results_csv, ground_truth_csv=None):
    """Calculate accuracy metrics for DDXPlus results"""
    
    # Load results
    df = pd.read_csv(results_csv)
    df = df.head(100)
    print(f"\n{'='*80}")
    print(f"Analyzing: {results_csv}")
    print(f"{'='*80}\n")
    
    # If True Diagnosis column is mostly empty, try to merge with ground truth
    if df['True Diagnosis'].isna().sum() > len(df) * 0.5:
        print("⚠️  True Diagnosis column mostly empty, attempting to merge with ground truth...")
        
        # Try to find ground truth file
        if ground_truth_csv is None:
            # Try default locations
            for gt_path in ['./dataset/AI_Data_Set_with_Categories_test.csv',
                          './dataset/AI_Data_Set_with_Categories_val.csv',
                          './dataset/AI_Data_Set_with_Categories_train.csv',
                          './dataset/AI_Data_Set_with_Categories_all.csv']:
                if os.path.exists(gt_path):
                    ground_truth_csv = gt_path
                    break
        
        if ground_truth_csv and os.path.exists(ground_truth_csv):
            print(f"   Using ground truth: {ground_truth_csv}")
            gt = pd.read_csv(ground_truth_csv, encoding='ISO-8859-1')
            
            # Merge to get ground truth
            df = df.merge(gt[['Participant No.', 'Processed Diagnosis', 'Level 2', 'Level 1']], 
                         on='Participant No.', how='left', suffixes=('', '_from_gt'))
            
            # Update True Diagnosis columns if they're empty
            if 'Processed Diagnosis' in df.columns:
                df['True Diagnosis'] = df['True Diagnosis'].fillna(df['Processed Diagnosis'])
            if 'Level 2' in df.columns and 'True Level 2' in df.columns:
                df['True Level 2'] = df['True Level 2'].fillna(df['Level 2'])
            if 'Level 1' in df.columns and 'True Level 1' in df.columns:
                df['True Level 1'] = df['True Level 1'].fillna(df['Level 1'])
            
            print(f"   ✓ Merged with ground truth: {df['True Diagnosis'].notna().sum()} rows with GT\n")
        else:
            print(f"   ❌ Could not find ground truth file\n")
    
    # Basic stats
    total_cases = len(df)
    has_generated = df['Generated Diagnosis'].notna().sum()
    missing = total_cases - has_generated
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total cases: {total_cases}")
    print(f"   With predictions: {has_generated} ({has_generated/total_cases*100:.1f}%)")
    print(f"   Missing predictions: {missing} ({missing/total_cases*100:.1f}%)")
    
    # Top-1 Accuracy (exact match)
    df['Generated_Clean'] = df['Generated Diagnosis'].apply(clean_diagnosis_name)
    df['True_Clean'] = df['True Diagnosis'].apply(clean_diagnosis_name)
    
    # Filter out rows with missing predictions
    df_valid = df[df['Generated_Clean'].notna() & (df['Generated_Clean'] != '')]
    
    exact_match = (df_valid['Generated_Clean'] == df_valid['True_Clean']).sum()
    top1_accuracy = exact_match / len(df_valid) if len(df_valid) > 0 else 0
    
    print(f"\n🎯 Top-1 Accuracy (Exact Match):")
    print(f"   Correct: {exact_match}/{len(df_valid)} = {top1_accuracy*100:.2f}%")
    
    # Flexible Accuracy (handles variations, aliases, substrings)
    df_valid = df_valid.copy()  # Avoid SettingWithCopyWarning
    df_valid['Flexible_Match'] = df_valid.apply(flexible_match, axis=1)
    flexible_match_count = df_valid['Flexible_Match'].sum()
    flexible_accuracy = flexible_match_count / len(df_valid) if len(df_valid) > 0 else 0
    
    print(f"\n🔀 Flexible Accuracy (Variations & Aliases):")
    print(f"   Correct: {flexible_match_count}/{len(df_valid)} = {flexible_accuracy*100:.2f}%")
    print(f"   Improvement: +{flexible_match_count - exact_match} cases ({flexible_accuracy*100 - top1_accuracy*100:.2f} pp)")
    
    # Level 1 and Level 2 accuracy (if available)
    if 'True Level 2' in df.columns and df['True Level 2'].notna().sum() > 0:
        level2_match = (df_valid['Generated_Clean'] == df_valid['True Level 2'].apply(clean_diagnosis_name)).sum()
        level2_accuracy = level2_match / len(df_valid) if len(df_valid) > 0 else 0
        print(f"\n📂 Level 2 Accuracy (Category Match):")
        print(f"   Correct: {level2_match}/{len(df_valid)} = {level2_accuracy*100:.2f}%")
    
    if 'True Level 1' in df.columns and df['True Level 1'].notna().sum() > 0:
        level1_match = (df_valid['Generated_Clean'] == df_valid['True Level 1'].apply(clean_diagnosis_name)).sum()
        level1_accuracy = level1_match / len(df_valid) if len(df_valid) > 0 else 0
        print(f"\n🏥 Level 1 Accuracy (System Match):")
        print(f"   Correct: {level1_match}/{len(df_valid)} = {level1_accuracy*100:.2f}%")
    
    # Confusion analysis
    print(f"\n❌ Most Common Misclassifications:")
    df_wrong = df_valid[df_valid['Generated_Clean'] != df_valid['True_Clean']]
    if len(df_wrong) > 0:
        confusion = df_wrong.groupby(['True Diagnosis', 'Generated Diagnosis']).size().sort_values(ascending=False).head(10)
        for (true_diag, pred_diag), count in confusion.items():
            print(f"   {true_diag} → {pred_diag}: {count} times")
    else:
        print("   (Perfect accuracy!)")
    
    # Per-disease accuracy
    print(f"\n📋 Per-Disease Performance (Top 10 by frequency):")
    disease_stats = []
    for disease in df_valid['True Diagnosis'].value_counts().index:
        disease_df = df_valid[df_valid['True Diagnosis'] == disease]
        correct = (disease_df['Generated_Clean'] == disease_df['True_Clean']).sum()
        total = len(disease_df)
        accuracy = correct / total if total > 0 else 0
        disease_stats.append({
            'Disease': disease,
            'Total': total,
            'Correct': correct,
            'Accuracy': f"{accuracy*100:.1f}%"
        })
    
    disease_df = pd.DataFrame(disease_stats)
    print(disease_df.to_string(index=False))
    
    # Bug checks
    print(f"\n🐛 Bug Checks:")
    if 'KG Info' in df.columns:
        localized_edema_count = df['KG Info'].str.contains('Localized edema', case=False, na=False).sum()
        print(f"   Rows with 'Localized Edema' in KG Info: {localized_edema_count}/{len(df)} ({localized_edema_count/len(df)*100:.1f}%)")
    
    anemia_gen_count = df['Generated Diagnosis'].str.contains('Anemia', case=False, na=False).sum()
    print(f"   Rows with 'Anemia' as Generated Diagnosis: {anemia_gen_count}/{len(df)} ({anemia_gen_count/len(df)*100:.1f}%)")
    
    # Sample KG Info (if available)
    if 'KG Info' in df.columns and df['KG Info'].notna().sum() > 0:
        print(f"\n📋 Sample KG Info Content (first 3 rows with KG Info):")
        pd.set_option('display.max_colwidth', 200)
        sample_df = df[df['KG Info'].notna()].head(3)
        if len(sample_df) > 0:
            display_cols = ['True Diagnosis', 'Generated Diagnosis', 'KG Info']
            # Only show columns that exist
            display_cols = [col for col in display_cols if col in sample_df.columns]
            print(sample_df[display_cols].to_string(index=False))
        pd.reset_option('display.max_colwidth')
    
    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"   {'='*60}")
    print(f"   Top-1 Accuracy (Exact): {top1_accuracy*100:.2f}%")
    print(f"   Flexible Accuracy:      {flexible_accuracy*100:.2f}%")
    print(f"   Success Rate:           {has_generated/total_cases*100:.1f}%")
    print(f"   Average Confidence:     N/A (not in results)")
    print(f"   {'='*60}")
    
    return {
        'total_cases': total_cases,
        'valid_predictions': len(df_valid),
        'top1_accuracy': top1_accuracy,
        'flexible_accuracy': flexible_accuracy,
        'exact_matches': exact_match,
        'flexible_matches': flexible_match_count
    }

def compare_results(result_files):
    """Compare multiple result files"""
    
    print(f"\n{'='*80}")
    print(f"Comparing {len(result_files)} result files")
    print(f"{'='*80}\n")
    
    comparison = []
    
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            # Calculate basic metrics
            total = len(df)
            df['Generated_Clean'] = df['Generated Diagnosis'].apply(clean_diagnosis_name)
            df['True_Clean'] = df['True Diagnosis'].apply(clean_diagnosis_name)
            df_valid = df[df['Generated_Clean'].notna() & (df['Generated_Clean'] != '')].copy()
            
            exact_match = (df_valid['Generated_Clean'] == df_valid['True_Clean']).sum()
            accuracy = exact_match / len(df_valid) if len(df_valid) > 0 else 0
            
            # Flexible accuracy
            df_valid['Flexible_Match'] = df_valid.apply(flexible_match, axis=1)
            flexible_match_count = df_valid['Flexible_Match'].sum()
            flexible_accuracy = flexible_match_count / len(df_valid) if len(df_valid) > 0 else 0
            
            # Try to extract config
            config_path = Path(result_file).parent / 'config.json'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            comparison.append({
                'Run': Path(result_file).parent.name,
                'File': Path(result_file).name,
                'Model': config.get('model', 'unknown'),
                'Total': total,
                'Valid': len(df_valid),
                'Exact': exact_match,
                'Exact Acc': f"{accuracy*100:.2f}%",
                'Flexible': flexible_match_count,
                'Flex Acc': f"{flexible_accuracy*100:.2f}%",
                'TopK': config.get('topk', '?'),
                'TopN': config.get('topn', '?'),
            })
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
    
    if comparison:
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description='Analyze MedRAG results on DDXPlus')
    parser.add_argument('inputs', nargs='+', help='Results CSV file(s) or directory containing results')
    parser.add_argument('--ground-truth', help='Ground truth CSV (optional)', default=None)
    parser.add_argument('--compare', action='store_true', help='Force comparison mode')
    
    args = parser.parse_args()
    
    # If multiple files provided, compare them
    if len(args.inputs) > 1:
        compare_results(args.inputs)
        return

    input_path = Path(args.inputs[0])
    
    if input_path.is_dir():
        # Find all CSV files in directory
        result_files = list(input_path.glob('*.csv'))
        if not result_files:
            # Check subdirectories
            result_files = list(input_path.glob('*/results_*.csv'))
            # Sort by timestamp (directory name)
            result_files.sort(key=lambda x: x.parent.name)
        
        if args.compare:
            compare_results(result_files)
        else:
            # Analyze each file separately
            for result_file in result_files:
                calculate_accuracy(result_file, args.ground_truth)
    
    elif input_path.is_file():
        calculate_accuracy(input_path, args.ground_truth)
    
    else:
        print(f"Error: {input_path} not found")

if __name__ == "__main__":
    main()

