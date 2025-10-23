#!/bin/bash
#SBATCH --job-name=atc_days
# SBATCH --output=/home/%u/.out/%j.out       ## this is where print() etc. go -> $HOME/.out
# SBATCH --error=/home/%u/.out/%j.err        ## this is where errors go       -> $HOME/.out
#SBATCH --time=0-24:00:00                   ## max. time in format d-hh:mm:ss
#SBATCH --nodes=1                           ## number of nodes, usually 1 in python
#SBATCH --mem-per-cpu=500MB                 ## specify the memory per core
# #SBATCH --mem=500MB                       ## alternatively, specify the memory (commented)
#SBATCH --ntasks=1                          ## number of tasks, usually 1 in python
#SBATCH --cpus-per-task=20                  ## number of cores
#SBATCH --partition=defq
# #SBATCH --account=my_special_project      ## account to charge the job to (commented)
#SBATCH --array=0-1%1   # number of days-1 % number of concurrent days 

set -euo pipefail
    
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

BASE_DIR="/store/kruu/atc_muac/audio_sdrplay"
OUT_DIR="./outputs"
SECTORS="delta_high"

START_DAY="2025-10-18"  # inclusive
DAY=$(date -u -d "${START_DAY} + ${SLURM_ARRAY_TASK_ID} day" +%Y-%m-%d)
START="${DAY}T00:00:00"
STOP=$(date -u -d "${DAY} + 1 day" +%Y-%m-%dT%H:%M:%S)

# load relevant module
module load mamba

echo "I am running on $SLURM_JOB_NODELIST"
echo "I am running with job id $SLURM_JOB_ID"

# run python script
cd "$APP_ROOT"
uv run -m core.run_pipeline \
  --base-dir "$BASE_DIR" \
  --sectors $SECTORS \
  --min-fl-m 7000 \
  --start-utc "$START" \
  --stop-utc  "$STOP" \
  --out-dir "$OUT_DIR" \
  --segmenter webrtc \
  --workers 128 \
  --show-progress
