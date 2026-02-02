#!/usr/bin/env python3
"""
Fix edge IDs in hainan/edges.shp by replacing all-zero key values with FID.
This is required for FMM to load the network properly.
"""

from osgeo import ogr
import sys

def fix_edge_ids(shapefile_path):
    """Replace all-zero key values with feature FID."""
    # Open the shapefile for editing
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(shapefile_path, 1)  # 1 for read/write

    if data_source is None:
        print(f"Error: Could not open {shapefile_path}")
        return False

    layer = data_source.GetLayer()
    feature_count = layer.GetFeatureCount()
    print(f"Total features: {feature_count}")

    # Get the index of the 'key' field
    key_field_index = layer.GetFeature(0).GetFieldIndex('key')
    if key_field_index == -1:
        print("Error: 'key' field not found")
        data_source = None
        return False

    print(f"Key field index: {key_field_index}")

    # Update each feature's key field with its FID
    layer.ResetReading()
    updated_count = 0
    for fid in range(feature_count):
        feature = layer.GetFeature(fid)
        if feature:
            old_key = feature.GetField('key')
            feature.SetField('key', fid)
            layer.SetFeature(feature)
            updated_count += 1

            if (updated_count - 1) % 10000 == 0:
                print(f"Updated {updated_count}/{feature_count} features (FID {fid}, old key: {old_key})")

    # Cleanup
    data_source = None
    print(f"Successfully updated {updated_count} features")
    return True

if __name__ == '__main__':
    shapefile = '/home/dell/fmm_sjtugnc/input/map/hainan/edges.shp'
    if fix_edge_ids(shapefile):
        print("Edge IDs fixed successfully!")
        sys.exit(0)
    else:
        print("Failed to fix edge IDs")
        sys.exit(1)
