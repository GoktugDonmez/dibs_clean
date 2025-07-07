#!/bin/bash -l
#SBATCH --job-name=dibs_experiments
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH --array=0-15
#SBATCH --output=logs/dibs_%A_%a.out   # %A=job-ID, %a=array-index
#SBATCH --error=logs/dibs_%A_%a.err

# set the environment
module load mamba
source activate dibs_env

# Define the combinations of parameters

grad_z_score_func_gmats=("hard" "soft")
grad_z_estimator_gmats=("hard" "soft")
grad_theta_score_func_gmats=("hard" "soft")
grad_theta_estimator_gmats=("hard" "soft")

# Calculate the combination for the current array job
gz_sf_idx=$((SLURM_ARRAY_TASK_ID % 2))
gz_e_idx=$(((SLURM_ARRAY_TASK_ID / 2) % 2))
gt_sf_idx=$(((SLURM_ARRAY_TASK_ID / 4) % 2))
gt_e_idx=$(((SLURM_ARRAY_TASK_ID / 8) % 2))

grad_z_score_func_gmat=${grad_z_score_func_gmats[$gz_sf_idx]}
grad_z_estimator_gmat=${grad_z_estimator_gmats[$gz_e_idx]}
grad_theta_score_func_gmat=${grad_theta_score_func_gmats[$gt_sf_idx]}
grad_theta_estimator_gmat=${grad_theta_estimator_gmats[$gt_e_idx]}

# Activate conda environment if needed
# source activate your_env

# Run the python script
srun python debug/dibs_experiment.py \
    --grad_z_score_func_gmat $grad_z_score_func_gmat \
    --grad_z_estimator_gmat $grad_z_estimator_gmat \
    --grad_theta_score_func_gmat $grad_theta_score_func_gmat \
    --grad_theta_estimator_gmat $grad_theta_estimator_gmat
