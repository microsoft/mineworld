import argparse
import os 
import json
import sys
import pandas as pd
import numpy as np
from rich import print

def tabluate_metrics(input_dir,output_path):
    metrics_list = []
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    idm_results = [f for f in all_files if 'idm' in f]
    fvd_results = [f for f in all_files if 'fvd' in f]
    exps = set([i.replace("idm_","").replace(".json","") for i in idm_results]) & set([i.replace("fvd_","").replace(".json","") for i in fvd_results])
    exps = list(exps)
    print(f"[bold magenta][Tabulating Evaluation Results][/bold magenta]: Found experiments : {exps}")
    for exp in exps:
        idm_file = os.path.join(input_dir, f"idm_{exp}.json")
        fvd_file = os.path.join(input_dir, f"fvd_{exp}.json")
        with open(idm_file, 'r') as f:
            idm_data = json.load(f)
        with open(fvd_file, 'r') as f:
            fvd_data = json.load(f)
        fvd_data = fvd_data["mean"]
        fvd_data.pop("exp_name", None)
        idm_data = idm_data["metric_mean_on_task"]
        metrics_entry = {
            "experiment": exp,
        }
        # merge dict 
        metrics_entry.update(fvd_data)
        metrics_entry.update(idm_data)
        metrics_list.append(metrics_entry)
    # Convert list of metrics to a DataFrame
    df = pd.DataFrame(metrics_list)

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"[bold red][Tabulating Evaluation Results End][/bold red] Metrics tabulated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate metrics from JSON files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSON metric files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the tabulated metrics CSV file")
    args = parser.parse_args()
    
    tabluate_metrics(args.input_dir, args.output_path)