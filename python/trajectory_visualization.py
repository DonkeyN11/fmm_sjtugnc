#!/usr/bin/env python3
"""
Trajectory Visualization Script
Visualizes matched trajectories from output/mr.txt on the Haikou road network
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


def load_trajectories(mr_file):
    """Load matched trajectories from mr.txt file"""
    print(f"Loading trajectories from {mr_file}...")

    # Read CSV file
    df = pd.read_csv(mr_file, sep=';')

    # Filter out empty trajectories
    df_valid = df[df['mgeom'] != 'LINESTRING()'].copy()

    if len(df_valid) == 0:
        print("No valid trajectories found!")
        return None

    print(f"Found {len(df_valid)} valid trajectories")

    # Parse geometry from mgeom column
    geometries = []
    for geom_str in df_valid['mgeom']:
        # Extract coordinates from LINESTRING(...) format
        if geom_str.startswith('LINESTRING('):
            coords_str = geom_str[11:-1]  # Remove 'LINESTRING(' and ')'
            coords = []
            for coord_pair in coords_str.split(','):
                if coord_pair.strip():
                    lon, lat = map(float, coord_pair.strip().split())
                    coords.append((lon, lat))
            if len(coords) > 1:
                geometries.append(LineString(coords))
            else:
                geometries.append(None)
        else:
            geometries.append(None)

    df_valid['geometry'] = geometries

    # Remove rows with None geometry
    df_valid = df_valid[df_valid['geometry'].notna()]

    return df_valid


def visualize_trajectories(mr_file, edges_shp, output_image='trajectory_visualization.png',
                          max_trajectories=100, sample_rate=0.1):
    """
    Visualize trajectories on the road network

    Parameters:
    - mr_file: Path to matched trajectory file
    - edges_shp: Path to edges shapefile
    - output_image: Output image filename
    - max_trajectories: Maximum number of trajectories to display
    - sample_rate: Sampling rate for large trajectory sets
    """

    # Load trajectories
    trajectories_df = load_trajectories(mr_file)
    if trajectories_df is None:
        return

    # Load road network
    print("Loading road network...")
    edges_gdf = gpd.read_file(edges_shp)
    print(f"Loaded {len(edges_gdf)} road segments")

    # Sample trajectories if too many
    if len(trajectories_df) > max_trajectories:
        n_sample = min(max_trajectories, int(len(trajectories_df) * sample_rate))
        trajectories_df = trajectories_df.sample(n=n_sample, random_state=42)
        print(f"Sampled {len(trajectories_df)} trajectories for visualization")

    # Create GeoDataFrame for trajectories
    trajectories_gdf = gpd.GeoDataFrame(trajectories_df, geometry='geometry', crs=edges_gdf.crs)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Plot road network (light gray background)
    edges_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.6, label='Road Network')

    # Plot trajectories with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories_gdf)))

    for i, (idx, row) in enumerate(trajectories_gdf.iterrows()):
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            if len(coords) > 1:
                # Create line segments
                segments = []
                for j in range(len(coords) - 1):
                    segments.append([coords[j], coords[j + 1]])

                # Create line collection
                lc = LineCollection(segments, colors=[colors[i]], linewidths=2, alpha=0.8)
                ax.add_collection(lc)

    # Set plot properties
    ax.set_title(f'Matched Trajectories Visualization\n{len(trajectories_gdf)} trajectories displayed',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=0, vmax=len(trajectories_gdf)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Trajectory Index', rotation=270, labelpad=15)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_image}")

    # Also create a summary plot
    create_summary_plot(trajectories_gdf, edges_gdf, output_image.replace('.png', '_summary.png'))

    plt.show()


def create_summary_plot(trajectories_gdf, edges_gdf, output_image):
    """Create a summary plot with trajectory statistics"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: All trajectories overview
    edges_gdf.plot(ax=ax1, color='lightgray', linewidth=0.3, alpha=0.5)
    for idx, row in trajectories_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            x, y = row.geometry.xy
            ax1.plot(x, y, 'b-', linewidth=0.5, alpha=0.6)
    ax1.set_title('All Trajectories Overview')
    ax1.set_aspect('equal')

    # Plot 2: Trajectory length distribution
    lengths = []
    for idx, row in trajectories_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            lengths.append(row.geometry.length)

    if lengths:
        ax2.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Trajectory Length Distribution')
        ax2.set_xlabel('Length (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Trajectory density heatmap
    all_coords = []
    for idx, row in trajectories_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            all_coords.extend(list(row.geometry.coords))

    if all_coords:
        coords_array = np.array(all_coords)
        ax3.hexbin(coords_array[:, 0], coords_array[:, 1], gridsize=50, cmap='hot', alpha=0.7)
        ax3.set_title('Trajectory Density Heatmap')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_aspect('equal')

    # Plot 4: Sample trajectories with different colors
    sample_size = min(10, len(trajectories_gdf))
    sample_indices = np.random.choice(len(trajectories_gdf), sample_size, replace=False)
    colors = plt.cm.Set3(np.linspace(0, 1, sample_size))

    edges_gdf.plot(ax=ax4, color='lightgray', linewidth=0.3, alpha=0.5)
    for i, idx in enumerate(sample_indices):
        row = trajectories_gdf.iloc[idx]
        if isinstance(row.geometry, LineString):
            x, y = row.geometry.xy
            ax4.plot(x, y, color=colors[i], linewidth=2, alpha=0.8,
                    label=f'Trajectory {row["id"]}')

    ax4.set_title(f'Sample Trajectories ({sample_size} shown)')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to {output_image}")


def main():
    """Main function"""
    # File paths
    mr_file = '/home/dell/Czhang/fmm_sjtugnc/output/mr.txt'
    edges_shp = '/home/dell/Czhang/fmm_sjtugnc/input/map/haikou/edges.shp'

    print("Trajectory Visualization Script")
    print("=" * 50)

    try:
        visualize_trajectories(mr_file, edges_shp, max_trajectories=50)
        print("Visualization completed successfully!")

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()