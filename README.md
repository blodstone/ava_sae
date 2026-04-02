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