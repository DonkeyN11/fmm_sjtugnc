import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
from pathlib import Path
from pyproj import Transformer
from sklearn.metrics import roc_curve, auc

def extract_coords_from_pgeom(df):
    """
    Extract lon/lat from pgeom column (e.g., 'POINT(110.123 20.123)').
    """
    if 'pgeom' not in df.columns:
        return df
    extracted = df['pgeom'].str.extract(r'POINT\s*\(\s*([\d\.\-]+)\s+([\d\.\-]+)\s*\)')
    df['matched_lon'] = extracted[0].astype(float)
    df['matched_lat'] = extracted[1].astype(float)
    return df.dropna(subset=['matched_lon', 'matched_lat'])

def calculate_projected_error(lon1, lat1, lon2, lat2, transformer):
    """
    Project lon/lat to meters and compute Euclidean distance.
    """
    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)
    mask = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
    errors = np.full(lon1.shape, np.nan)
    errors[mask] = np.sqrt((x1[mask] - x2[mask])**2 + (y1[mask] - y2[mask])**2)
    return errors

def process_method_data(df_result, df_input, threshold, transformer):
    """
    Align data, calculate error, generate true labels and prediction scores.
    """
    # 1. Preprocess
    df_result = extract_coords_from_pgeom(df_result)
    
    # 2. Align
    merged = pd.merge(df_result, df_input[['id', 'timestamp', 'x', 'y']], 
                      on=['id', 'timestamp'], how='inner')
    
    if merged.empty:
        return None, None, merged

    # 3. Calculate Euclidean Error
    merged['error_m'] = calculate_projected_error(
        merged['x'].values, merged['y'].values, 
        merged['matched_lon'].values, merged['matched_lat'].values,
        transformer
    )
    
    # 4. Filter NaNs
    # Handle cases where 'trustworthiness' might be named differently or missing
    trust_col = 'trustworthiness' if 'trustworthiness' in merged.columns else None
    if not trust_col:
        print("Warning: 'trustworthiness' column missing in dataset.")
        return None, None, merged
        
    merged = merged.dropna(subset=['error_m', trust_col])
    
    # 5. Build ROC Labels and Scores
    # Positive class (1) is "Error"
    y_true = (merged['error_m'] > threshold).astype(int)
    
    # Lower trustworthiness means higher error probability -> Predict Score = 1 - trust
    y_score = 1.0 - merged[trust_col]
    
    return y_true, y_score, merged

def main():
    parser = argparse.ArgumentParser(description='Analyze Trustworthiness ROC for Error Detection.')
    parser.add_argument('-t', '--threshold', type=float, default=10.0,
                        help='Error threshold in meters to define "Matching Error" (default: 10.0m)')
    args = parser.parse_args()

    # --- 1. Path Configuration ---
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset-hainan-06"
    mr_dir = data_dir / "mr"
    output_dir = base_dir / "python_dataset" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = data_dir / "cmm_input_points.csv"
    cmm_csv = mr_dir / "cmm_results_0303_only_trust_norm.csv"
    fmm_csv = mr_dir / "fmm_results_filtered.csv"

    # --- 2. Initialize Transformer ---
    transformer = Transformer.from_crs("epsg:4326", "epsg:32649", always_xy=True)

    # --- 3. Data Loading ---
    print(f"Loading datasets... (Threshold: {args.threshold}m)")
    try:
        df_input = pd.read_csv(input_csv, sep=';')
        df_cmm = pd.read_csv(cmm_csv, sep=';')
        df_fmm = pd.read_csv(fmm_csv, sep=';')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- 4. Process ROC Data ---
    print("Processing CMM and FMM data for ROC analysis...")
    y_true_cmm, y_score_cmm, _ = process_method_data(df_cmm, df_input, args.threshold, transformer)
    y_true_fmm, y_score_fmm, _ = process_method_data(df_fmm, df_input, args.threshold, transformer)

    results = []
    
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid")

    if y_true_cmm is not None and len(np.unique(y_true_cmm)) > 1:
        fpr, tpr, _ = roc_curve(y_true_cmm, y_score_cmm)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CMM Trustworthiness (AUC = {roc_auc:.3f})')
        print(f"CMM AUC: {roc_auc:.4f}")
    else:
        print("CMM: Not enough class diversity or data to compute ROC.")

    if y_true_fmm is not None and len(np.unique(y_true_fmm)) > 1:
        fpr, tpr, _ = roc_curve(y_true_fmm, y_score_fmm)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='royalblue', lw=2, label=f'FMM Trustworthiness (AUC = {roc_auc:.3f})')
        print(f"FMM AUC: {roc_auc:.4f}")
    else:
        print("FMM: Not enough class diversity or data to compute ROC.")

    # Reference line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)\n(Incorrectly flagged as Error)')
    plt.ylabel('True Positive Rate (TPR)\n(Successfully detected Error)')
    plt.title(f'ROC Curve: Trustworthiness as Error Detector\n(Error > {args.threshold}m as True Label)')
    plt.legend(loc="lower right")
    
    img_path = output_dir / f'trustworthiness_roc_{args.threshold}m.png'
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    print(f"ROC plot saved to {img_path}")

if __name__ == "__main__":
    main()
