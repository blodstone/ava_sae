# Do Language Models Encode Knowledge of Linguistic Constraint Violations?
This repository contains the source code and data for our paper: "Do Language Models Encode Knowledge of Linguistic Constraint Violations?"

# Citation
TBA

# Setup Instructions
1. Install the dependency packages `uv sync`
1. Activate the virtual environment `source .venv/bin/activate`
1. Clone this repo: git clone (Todo)
1. Clone BLiMP repo: `cd data && git clone https://github.com/alexwarstadt/blimp`
1. Preprocess datasets: `cd ../src && python preprocess_data.py --dataset -all`

# Decoupling of formal and functional linguistics 
This is the source code for the decoupling of formal and functional linguistics project.

## Evaluating BLIMP and FOLIO datasets using unpatched language models
Download BLIMP dataset. 
```bash
python evaluate_blimp.py --blimp_path \[blimp dir\] --output_path blimp_results.jsonl 
```

Download FOLIO dataset
```bash
python evaluate_folio.py --model_name gpt2 --folio_path /home/users1/hardy/hardy/datasets/FOLIO/folio_v2_validation.jsonl --output_path folio_results_gpt2.jsonl --batch_size 8
```