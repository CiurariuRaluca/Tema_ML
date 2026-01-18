import os
import pandas as pd
import numpy as np


def save_model_coefficients(model, feature_names, model_name, out_dir="../../results"):

    os.makedirs(out_dir, exist_ok=True)

    coef_df = pd.DataFrame({"feature": feature_names, "coef": model.W})

    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["abs_coef"] = np.abs(coef_df["coef"])

    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    filename = os.path.join(out_dir, f"coefficients_{model_name}.csv")
    coef_df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved: {filename}")
