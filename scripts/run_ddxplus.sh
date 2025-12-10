#!/bin/bash
#SBATCH --job-name=paper-KG         # Job name
#SBATCH --account=eecs542f25_class                 # account
#SBATCH --partition=spgpu               # Specify the A40 GPU partition or queue
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem-per-gpu=24g             # Memory per GPU
#SBATCH --time=08:00:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=output/run_ddxplus/paper-KG.%j.out    # Output file
#SBATCH --error=output/run_ddxplus/paper-KG.%j.err     # Error file

# Activate the conda environment
source ~/.bashrc
conda activate medrag

# run the python script
python run_ddxplus.py --kg-path ./dataset/medrag_kg_ddxplus.xlsx --max-samples 500
# python run_ddxplus.py --kg-path ./dataset/llm_kg/knowledge_graph_DDXPlus_LLM.xlsx --max-samples 100
# python run_ddxplus.py --kg-path ./dataset/llm_kg_v2/knowledge_graph_DDXPlus_LLM_v2.xlsx --max-samples 500
# print the end time
echo "Job completed at $(date)"
