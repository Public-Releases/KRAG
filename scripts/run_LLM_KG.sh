#!/bin/bash
#SBATCH --job-name=MedRAG         # Job name
#SBATCH --account=mni0                 # account
#SBATCH --partition=spgpu               # Specify the A40 GPU partition or queue
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem-per-gpu=24g             # Memory per GPU
#SBATCH --time=08:00:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=output/sample_sbatch_job.%j.out    # Output file
#SBATCH --error=output/sample_sbatch_job.%j.err     # Error file

# Activate the conda environment
source ~/.bashrc
conda activate medrag

# run the python script
python run_ddxplus.py --kg-path ./dataset/llm_kg/knowledge_graph_DDXPlus_LLM.xlsx --max-samples 500

# print the end time
echo "Job completed at $(date)"
