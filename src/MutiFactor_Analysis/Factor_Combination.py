import os
import pandas as pd

# Combine and rename factor files according to merge_dict
def merge_and_rename_factors(input_folder, output_folder, merge_dict=None):
    os.makedirs(output_folder, exist_ok=True)

    # Load all _final.csv factor files
    all_factors = {
        fname.replace("_final.csv", ""): pd.read_csv(
            os.path.join(input_folder, fname), index_col=0, parse_dates=True
        )
        for fname in os.listdir(input_folder)
        if fname.endswith("_final.csv")
    }

    used_factors = set()

    # Merge according to merge_dict
    if merge_dict:
        for new_name, factor_list in merge_dict.items():
            merged = [
                all_factors[f] for f in factor_list if f in all_factors
            ]
            if merged:
                combined_df = sum(merged) / len(merged)
                combined_df.to_csv(os.path.join(output_folder, f"{new_name}_combined.csv"))
                used_factors.update(factor_list)

    # Save unused factors
    for name, df in all_factors.items():
        if name not in used_factors:
            df.to_csv(os.path.join(output_folder, f"{name}_combined.csv"))

merge_dict = {
    "safety_factors": [
        "CR_safety_factors", 
        "CUR_safety_factors", 
        "QR_safety_factors"
    ]
}

merge_and_rename_factors(
    input_folder="/Users/a12205/Desktop/美国实习/INS",
    output_folder="/Users/a12205/Desktop/美国实习/INS",
    merge_dict=merge_dict
)
merge_and_rename_factors(
    input_folder="/Users/a12205/Desktop/美国实习/OOS",
    output_folder="/Users/a12205/Desktop/美国实习/OOS",
    merge_dict=merge_dict
)
