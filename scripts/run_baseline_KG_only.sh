#!/bin/bash
#SBATCH --job-name=paper-KG-ONLY         # Job name
#SBATCH --account=mni0                 # account
#SBATCH --partition=spgpu               # Specify the A40 GPU partition or queue
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem-per-gpu=24g             # Memory per GPU
#SBATCH --time=08:00:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=output/run_ablation/paper-KG-ONLY.%j.out    # Output file
#SBATCH --error=output/run_ablation/paper-KG-ONLY.%j.err     # Error file

# Activate the conda environment
source ~/.bashrc
conda activate medrag


# run the python script
python run_ablation.py --mode kg_only --max_samples 500 --split test --model local/llama-3.1-8b --kg-path ./dataset/medrag_kg_ddxplus.xlsx
# python run_ablation.py --mode kg_only --max_samples 500 --split test --model local/llama-3.1-8b --kg-path ./dataset/llm_kg_v2/knowledge_graph_DDXPlus_LLM_v2.xlsx

# print the end time
echo "Job completed at $(date)"
