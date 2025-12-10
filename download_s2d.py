from datasets import load_dataset
import pandas as pd

test_df = pd.read_csv("symptom_to_diagnosis_test.csv")

train_df = pd.read_csv("symptom_to_diagnosis_train.csv")


target_ids = ["bronchial asthma", "common cold", "gastroesophageal reflux disease", "pneumonia"]
filtered_test_df = test_df[test_df['output_text'].isin(target_ids)].copy()
filtered_train_df = train_df[train_df['output_text'].isin(target_ids)].copy()



mapping = {
    "bronchial asthma": "Bronchospasm / acute asthma exacerbation",
    "common cold": "URTI",
    "gastroesophageal reflux disease": "GERD",
    "pneumonia": "Pneumonia"
}
filtered_test_df['true_diagnosis_ddxplus'] = filtered_test_df['output_text'].map(mapping)
filtered_train_df['true_diagnosis_ddxplus'] = filtered_train_df['output_text'].map(mapping)

filtered_test_df.to_csv("symptom_to_diagnosis_test_filtered.csv", index=False)
filtered_train_df.to_csv("symptom_to_diagnosis_train_filtered.csv", index=False)
