#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition bosch_cpu-cascadelake

# Define a name for your job
#SBATCH --job-name Metadata

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A.out

# Define the amount of memory required per node
#SBATCH --mem=48GB
#SBATCH -c 8
#SBATCH --gres=localtmp:100

#Time Format = days-hours:minutes:seconds
#SBATCH --time=8-00:00:00

#SBATCH --propagate=NONE

#SBATCH --array=0-32  # Adjust based on the number of methods

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
# datasets=(146818 146820 168350 168911 190137 190411 359955 359956 359979)
datasets_regression=(359944 359929 233212 359937 359950 359938 233213 359942 233211 359936 359952 359951 359949 233215 360945 167210 359943 359941 359946 360933 360932 359930 233214 359948 359931 359932 359933 359934 359939 359945 359935 317614 359940) # regression
# datasets_classification=(190411 359983 189354 189356 10090 359979 168868 190412 146818 359982 359967 359955 359960 359973 359968 359992 359959 359957 359977 7593 168757 211986 168909 189355 359964 359954 168910 359976 359969 359970 189922 359988 359984 360114 359966 211979 168911 359981 359962 360975 3945 360112 359991 359965 190392 359961 359953 359990 359980 167120 359993 190137 359958 190410 359971 168350 360113 359956 359989 359986 359975 359963 359994 359987 168784 359972 190146 359985 146820 359974 2073) # classification
dataset_regression=${datasets_regression[$SLURM_ARRAY_TASK_ID]}
# dataset_classification=${datasets_classification[$SLURM_ARRAY_TASK_ID]}

# Running the job
# shellcheck disable=SC2006
start=`date +%s`

# shellcheck disable=SC2048
python3 src/Metadata/core/Operator_Model_Feature_Matrix_Regression_Parallel.py --dataset "$dataset_regression"
# python3 src/Metadata/core/Operator_Model_Feature_Matrix_Classification_Parallel.py --dataset "$dataset_classification"


# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

deactivate

# shellcheck disable=SC2006
end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"
