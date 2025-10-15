#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a GCJ-02 (Gaode) encoded OSM extract into WGS84 nodes/edges shapefiles.

Example:
    python convert_gaode_osm.py \
        --input input/map/hainan-gaode-init.osm \
        --output-dir input/map/hainan
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple, Union

import geopandas as gpd
import osmnx as ox
from pyproj import Geod
from shapely.geometry import base as shapely_base
from shapely.ops import transform as shapely_transform


GCJ_A = 6378245.0
GCJ_EE = 0.00669342162296594323
GEOD = Geod(ellps="WGS84")


def _transform_lat(x: float, y: float) -> float:
    # Standard GCJ-02 helper.
    ret = (
        -100.0
        + 2.0 * x
        + 3.0 * y
        + 0.2 * y * y
        + 0.1 * x * y
        + 0.2 * math.sqrt(abs(x))
    )
    ret += (
        (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0
        / 3.0
    )
    ret += (
        (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    )
    ret += (
        (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0))
        * 2.0
        / 3.0
    )
    return ret


def _transform_lon(x: float, y: float) -> float:
    # Standard GCJ-02 helper.
    ret = (
        300.0
        + x
        + 2.0 * y
        + 0.1 * x * x
        + 0.1 * x * y
        + 0.1 * math.sqrt(abs(x))
    )
    ret += (
        (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0
        / 3.0
    )
    ret += (
        (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    )
    ret += (
        (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi))
        * 2.0
        / 3.0
    )
    return ret


def _out_of_china(lat: float, lon: float) -> bool:
    return not (73.66 <= lon <= 135.05 and 3.86 <= lat <= 53.55)


def gcj02_to_wgs84(lon: float, lat: float) -> Tuple[float, float]:
    if _out_of_china(lat, lon):
        return lon, lat
    d_lat = _transform_lat(lon - 105.0, lat - 35.0)
    d_lon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - GCJ_EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((GCJ_A * (1 - GCJ_EE)) / (magic * sqrt_magic) * math.pi)
    d_lon = (d_lon * 180.0) / (GCJ_A / sqrt_magic * math.cos(rad_lat) * math.pi)
    mg_lat = lat + d_lat
    mg_lon = lon + d_lon
    return lon * 2 - mg_lon, lat * 2 - mg_lat


def _vectorized_gcj2wgs(
    x: Union[float, Iterable[float]],
    y: Union[float, Iterable[float]],
    z: Union[float, Iterable[float], None] = None,
):
    # Shapely <2 passes scalars, Shapely >=2 passes numpy arrays.
    try:
        len(x)  # type: ignore[arg-type]
    except TypeError:
        lon, lat = gcj02_to_wgs84(float(x), float(y))
        if z is None:
            return lon, lat
        return lon, lat, z

    # Iterable case.
    lon_list = []
    lat_list = []
    for xi, yi in zip(x, y):
        lon_i, lat_i = gcj02_to_wgs84(float(xi), float(yi))
        lon_list.append(lon_i)
        lat_list.append(lat_i)
    if z is None:
        return lon_list, lat_list
    return lon_list, lat_list, z


def convert_geometry(geom: shapely_base.BaseGeometry):
    if geom is None or geom.is_empty:
        return geom
    return shapely_transform(_vectorized_gcj2wgs, geom)


def geodesic_length_meters(geom: shapely_base.BaseGeometry) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if hasattr(GEOD, "geometry_length"):
        return float(GEOD.geometry_length(geom))
    geom_type = geom.geom_type
    if geom_type in {"LineString", "LinearRing"}:
        lon, lat = geom.xy
        return float(GEOD.line_length(lon, lat))
    if geom_type == "MultiLineString":
        return float(sum(geodesic_length_meters(part) for part in geom.geoms))
    return 0.0


def normalize_osmid(value):
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return value


def clear_shapefile(base_path: Path):
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        target = base_path.with_suffix(suffix)
        if target.exists():
            target.unlink()


def convert(osm_path: Path, output_dir: Path):
    print(f"Loading OSM graph from {osm_path} ...")
    graph = ox.graph_from_xml(
        filepath=str(osm_path),
        simplify=True,
        retain_all=True,
    )

    print("Converting graph to GeoDataFrames ...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True, fill_edge_geometry=True)

    print("Transforming coordinates from GCJ-02 to WGS84 ...")
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["geometry"] = nodes_gdf.geometry.apply(convert_geometry)
    nodes_gdf["x"] = nodes_gdf.geometry.x
    nodes_gdf["y"] = nodes_gdf.geometry.y
    nodes_gdf = nodes_gdf.set_crs(epsg=4326, allow_override=True)
    if "osmid" in nodes_gdf.columns:
        nodes_gdf["osmid"] = nodes_gdf["osmid"].apply(normalize_osmid)

    edges_gdf = edges_gdf.copy()
    edges_gdf = edges_gdf.reset_index(drop=False)
    edges_gdf = edges_gdf[edges_gdf.geometry.notnull()].reset_index(drop=True)
    edges_gdf["geometry"] = edges_gdf.geometry.apply(convert_geometry)
    edges_gdf = edges_gdf[~edges_gdf.geometry.is_empty].reset_index(drop=True)
    edges_gdf = edges_gdf.set_crs(epsg=4326, allow_override=True)
    if "osmid" in edges_gdf.columns:
        edges_gdf["osmid"] = edges_gdf["osmid"].apply(normalize_osmid)

    print("Recomputing edge lengths (meters) ...")
    edges_gdf["length"] = edges_gdf.geometry.apply(geodesic_length_meters)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.shp"
    edges_path = output_dir / "edges.shp"
    clear_shapefile(nodes_path)
    clear_shapefile(edges_path)

    print(f"Writing nodes shapefile to {nodes_path} ...")
    nodes_gdf.to_file(nodes_path, driver="ESRI Shapefile", encoding="utf-8")

    print(f"Writing edges shapefile to {edges_path} ...")
    edges_gdf.to_file(edges_path, driver="ESRI Shapefile", encoding="utf-8")

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GCJ-02 encoded OSM extract into WGS84 nodes/edges shapefiles."
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        required=True,
        help="Path to the source .osm or .osm.xml file in GCJ-02.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        required=True,
        help="Directory where nodes.shp and edges.shp will be written.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    osm_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not osm_path.exists():
        raise FileNotFoundError(f"Input file not found: {osm_path}")

    convert(osm_path, output_dir)


if __name__ == "__main__":
    main()
