#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=cpufast
#SBATCH --job-name claims_quality_metrics
#SBATCH --out=/home/ryparmar/logs/claims_quality_metrics.out

# Load modules
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
source /home/ryparmar/venv/docretr/bin/activate

# export DATA="/mnt/data/factcheck/CTK/dataset/v2.1/nfc"
# export DATA="/mnt/data/factcheck/CTK/par5/ctk-data"
# export DATA="/mnt/data/factcheck/fever/data-cs/fever-data"


# export DIR="/mnt/data/factcheck/CTK/dataset/dr_concise_latest"
# export SPLIT="fever-data"
# export DATA=${DIR}

export DIR="/mnt/data/factcheck/fever/fever-data-deepl/nfc"
export DATA=${DIR}

# export EXPORT="/home/ryparmar/tmp/tmp2/ctk-v2.1"
export EXPORT="/home/ryparmar/claim_quality_metrics_fever_cs_deepl"
export MORPHODITA="/mnt/data/factcheck/ufal/morphodita/czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger"
export DATASPLIT="all"

python3 src/main.py \
        --data ${DATA} \
        --cues unigram \
        --cv \
        --export ${EXPORT} \
        --split ${DATASPLIT} \
        --save_latex \

python3 src/main.py \
        --data ${DATA} \
        --cues unigram \
        --cv \
        --export ${EXPORT} \
        --split ${DATASPLIT} \
        --save_latex \
        --negation_only \
        --morphodita ${MORPHODITA}

python3 src/main.py \
        --data ${DATA} \
        --cues bigram \
        --cv \
        --export ${EXPORT} \
        --split ${DATASPLIT} \
        --save_latex

python3 src/main.py \
        --data ${DATA} \
        --cues wordpiece \
        --cv \
        --export ${EXPORT} \
        --split ${DATASPLIT} \
        --save_latex
        
