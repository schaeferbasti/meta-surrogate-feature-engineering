#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition bosch_cpu-cascadelake

# Define a name for your job
#SBATCH --job-name Metadata

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A.out
#SBATCH --error logs/%x-%A_%a.err    # STDERR  short: -e logs/%x-%A.out

# Define the amount of memory required per node
#SBATCH --mem=48GB
#SBATCH -c 8
#SBATCH --gres=localtmp:100

#Time Format = days-hours:minutes:seconds
#SBATCH --time=4-00:00:00

#SBATCH --propagate=NONE

#SBATCH --array=0-8  # Adjust based on the number of methods

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

# Activate your environment
# shellcheck disable=SC1090
# source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
# if conda info --envs | grep -q amltk_env; then echo "amltk_env already exists"; else conda create -y -n amltk_env; fi
# conda activate amltk_env
# echo "conda amltk_env activated"
# virtualenv .venv
source .venv/local/bin/activate
echo "Virtual Environment Activated"

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$PWD/src:$PYTHONPATH
echo "PYTHONPATH set to $PYTHONPATH"

# datasets=(190411 189354 189356 359979 146818 359955 359960 359968 359959 168757 359954 359969 359970 359984 168911 359981 359962 359965 190392 190137 359958 168350 359956 359975 359963 168784 190146 146820 359974 2073 359944 359950 359942 359951 360945 167210 359930 359948 359931 359932 359933 359934 359939 359945 359935 359940)
datasets=(146818 146820 168350 168911 190137 190411 359955 359956 359979)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

# Running the job
# shellcheck disable=SC2006
start=`date +%s`

# shellcheck disable=SC2048
python3 src/Metadata/Operator_Model_Feature_Matrix_parallel.py --dataset "$dataset"

# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

deactivate

# shellcheck disable=SC2006
end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"
