#!/bin/bash
#SBATCH --job-name=mp_py_benchmark
# SBATCH --output=/home/%u/.out/%j.out       ## this is where print() etc. go -> $HOME/.out
# SBATCH --error=/home/%u/.out/%j.err        ## this is where errors go       -> $HOME/.out
#SBATCH --time=0-24:00:00                   ## max. time in format d-hh:mm:ss
#SBATCH --nodes=1                           ## number of nodes, usually 1 in python
#SBATCH --mem-per-cpu=500MB                 ## specify the memory per core
# #SBATCH --mem=500MB                       ## alternatively, specify the memory (commented)
#SBATCH --ntasks=1                          ## number of tasks, usually 1 in python
#SBATCH --cpus-per-task=100                  ## number of cores
#SBATCH --partition=defq                    ## queue (partition) to run the job in
# #SBATCH --partition=qjupyter              ## alternative queue (commented)
# #SBATCH --nodelist=srv-lab-t-251          ## run on a specific worker (commented)exit
# #SBATCH --account=my_special_project      ## account to charge the job to (commented)

# create output directory (doesn't do anything if it already exists)
mkdir -p ${HOME}/.out

# set up environment variable with parent directory of this script
#
# Source: https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
#
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi
echo "SCRIPT_PATH: $SCRIPT_PATH"

APP_ROOT="$(dirname "$SCRIPT_PATH")"
echo "APP_ROOT: $APP_ROOT"

# load relevant module
module load mamba

# activate environment
# Note: You can activate any env that is installed in your home directory
# source activate aware

echo "I am running on $SLURM_JOB_NODELIST"
echo "I am running with job id $SLURM_JOB_ID"

# run python script in the activated environment
# -> make sure that the path matches your setup
uv run ${APP_ROOT}/utils/sftp_pull_sync.py
