import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from pathlib import Path
from pyproj import Transformer

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

def main():
    # --- 1. Path Configuration ---
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset-hainan-06"
    mr_dir = data_dir / "mr"
    output_dir = base_dir / "python_dataset" / "output_PL=10"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = data_dir / "cmm_input_points.csv"
    cmm_csv = mr_dir / "cmm_PL=10.csv"
    fmm_csv = mr_dir / "fmm_results_filtered.csv"

    for f in [input_csv, cmm_csv, fmm_csv]:
        if not f.exists():
            print(f"Error: File not found: {f}")
            return

    # --- 2. Initialize Projection Transformer ---
    transformer = Transformer.from_crs("epsg:4326", "epsg:32649", always_xy=True)

    # --- 3. Data Loading ---
    print(f"Loading datasets...")
    df_input = pd.read_csv(input_csv, sep=';')
    df_cmm = pd.read_csv(cmm_csv, sep=';')
    df_fmm = pd.read_csv(fmm_csv, sep=';')

    # --- 4. Preprocessing ---
    df_cmm = extract_coords_from_pgeom(df_cmm)
    df_fmm = extract_coords_from_pgeom(df_fmm)

    input_cols = ['id', 'timestamp', 'x', 'y']
    df_input_sub = df_input[input_cols].copy()

    # --- 5. Data Alignment & Error Calculation ---
    print("Aligning data and calculating Euclidean errors in meters...")
    
    # CMM Error
    cmm_merged = pd.merge(df_cmm, df_input_sub, on=['id', 'timestamp'], how='inner')
    if not cmm_merged.empty:
        cmm_merged['error_m'] = calculate_projected_error(
            cmm_merged['x'].values, cmm_merged['y'].values, 
            cmm_merged['matched_lon'].values, cmm_merged['matched_lat'].values,
            transformer
        )
        cmm_merged = cmm_merged.dropna(subset=['error_m'])

    # FMM Error
    fmm_merged = pd.merge(df_fmm, df_input_sub, on=['id', 'timestamp'], how='inner')
    if not fmm_merged.empty:
        fmm_merged['error_m'] = calculate_projected_error(
            fmm_merged['x'].values, fmm_merged['y'].values, 
            fmm_merged['matched_lon'].values, fmm_merged['matched_lat'].values,
            transformer
        )
        fmm_merged = fmm_merged.dropna(subset=['error_m'])

    if cmm_merged.empty or fmm_merged.empty:
        print("Error: No valid aligned data found.")
        return

    # --- 6. Visualization ---
    print("Generating plots (Error + Trustworthiness)...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Prepare combined data
    cmm_merged['Method'] = 'CMM'
    fmm_merged['Method'] = 'FMM'
    
    plot_cols = ['error_m', 'Method']
    if 'trustworthiness' in cmm_merged.columns: plot_cols.append('trustworthiness')
    
    plot_df = pd.concat([
        cmm_merged[[c for c in plot_cols if c in cmm_merged.columns]],
        fmm_merged[[c for c in plot_cols if c in fmm_merged.columns]]
    ], ignore_index=True)

    q99 = plot_df['error_m'].quantile(0.99)
    if pd.isna(q99) or q99 <= 0: q99 = 50.0

    # Subplot 1: PDF of Error
    sns.kdeplot(data=plot_df, x="error_m", hue="Method", fill=True, common_norm=False, ax=axes[0], cut=0)
    axes[0].set_title('Matching Error PDF')
    axes[0].set_xlim(0, q99)

    # Subplot 2: CDF of Error
    sns.ecdfplot(data=plot_df, x="error_m", hue="Method", ax=axes[1])
    axes[1].set_title('Matching Error CDF')
    axes[1].set_xlim(0, q99)
    axes[1].axhline(0.95, color='gray', linestyle='--')

    # Subplot 3: Box Plot of Error
    sns.boxplot(data=plot_df, x="Method", y="error_m", ax=axes[2], showfliers=False)
    axes[2].set_title('Error Distribution (No fliers)')

    # Subplot 4: Distribution of Trustworthiness
    if 'trustworthiness' in plot_df.columns:
        sns.histplot(data=plot_df, x="trustworthiness", hue="Method", element="step", common_norm=False, kde=True, ax=axes[3])
        axes[3].set_title('Trustworthiness Score Distribution')
        axes[3].set_xlabel('Trustworthiness (0.0 to 1.0)')
        
        print("\nTrustworthiness Stats:")
        for method in ['CMM', 'FMM']:
            m_data = plot_df[plot_df['Method'] == method]['trustworthiness']
            if not m_data.empty:
                print(f"  {method}: Mean={m_data.mean():.4f}, Median={m_data.median():.4f}, Std={m_data.std():.4f}")
    else:
        axes[3].text(0.5, 0.5, 'Trustworthiness column missing', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'matching_errors_and_trust_comparison.png', dpi=300)

    # --- 7. Comparative Outlier Analysis ---
    print("\nGenerating comparative analysis for large errors...")
    comparison = pd.merge(
        cmm_merged[['id', 'timestamp', 'error_m', 'trustworthiness', 'matched_lon', 'matched_lat']],
        fmm_merged[['id', 'timestamp', 'error_m']],
        on=['id', 'timestamp'], suffixes=('_cmm', '_fmm')
    )
    comparison['improvement_m'] = comparison['error_m_fmm'] - comparison['error_m_cmm']
    comparison = pd.merge(comparison, df_input_sub, on=['id', 'timestamp'])
    
    threshold = 10.0
    outliers = comparison[(comparison['error_m_cmm'] > threshold) | (comparison['error_m_fmm'] > threshold)].copy()
    outliers = outliers.sort_values(by='improvement_m', ascending=False)
    
    outlier_path = output_dir / 'large_error_analysis.csv'
    outliers.to_csv(outlier_path, index=False, sep=';', float_format='%.6f')
    print(f"Analysis saved to {outlier_path}")

if __name__ == "__main__":
    main()
