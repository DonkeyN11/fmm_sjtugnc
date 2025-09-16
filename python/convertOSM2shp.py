import osmnx as ox
import pdb
import traceback
import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import threading
import requests
from functools import wraps

def retry_on_connection_error(max_retries=3, delay=5):
    """装饰器：在连接错误时自动重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ProtocolError,
                        ConnectionResetError) as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # 指数退避
            return None
        return wrapper
    return decorator

class ProgressTracker:
    def __init__(self, total_steps=4):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.step_names = [
            "Converting graph to GeoDataFrames",
            "Stringifying non-numeric columns",
            "Processing edge IDs",
            "Saving shapefiles"
        ]

    def update(self, step_name=None):
        with self.lock:
            self.current_step += 1
            elapsed = time.time() - self.start_time
            progress = (self.current_step / self.total_steps) * 100

            if step_name:
                current_name = step_name
            elif self.current_step <= len(self.step_names):
                current_name = self.step_names[self.current_step - 1]
            else:
                current_name = f"Step {self.current_step}"

            print(f"[{progress:5.1f}%] {current_name}... (Elapsed: {elapsed:.1f}s)")

            if self.current_step == self.total_steps:
                print(f"[100.0%] Completed! Total time: {elapsed:.1f}s")

def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8", max_workers=None, show_progress=True):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # Initialize progress tracker
    progress = ProgressTracker() if show_progress else None

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    if progress:
        progress.update()

    # Use all available CPU cores if not specified
    if max_workers is None:
        max_workers = cpu_count()

    def save_gdf_task(gdf, filepath, encoding, task_name=""):
        """Helper function to save a GeoDataFrame to file"""
        if progress and show_progress:
            start_time = time.time()
        gdf.to_file(filepath, encoding=encoding)
        if progress and show_progress:
            elapsed = time.time() - start_time
            print(f"    └─ Saved {task_name} ({os.path.basename(filepath)}) in {elapsed:.1f}s")
        return filepath

    # Parallel processing for stringifying non-numeric columns
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        stringify_future_nodes = executor.submit(ox.io._stringify_nonnumeric_cols, gdf_nodes)
        stringify_future_edges = executor.submit(ox.io._stringify_nonnumeric_cols, gdf_edges)

        gdf_nodes = stringify_future_nodes.result()
        gdf_edges = stringify_future_edges.result()

    if progress:
        progress.update()

    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    if progress:
        progress.update("Processing edge IDs")

    # Parallel processing for saving shapefiles
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        save_nodes_future = executor.submit(save_gdf_task, gdf_nodes, filepath_nodes, encoding, "nodes")
        save_edges_future = executor.submit(save_gdf_task, gdf_edges, filepath_edges, encoding, "edges")

        # Wait for both operations to complete
        save_nodes_future.result()
        save_edges_future.result()

    if progress:
        progress.update("Saving shapefiles")

@retry_on_connection_error(max_retries=5, delay=10)
def download_graph_with_retry(place_name, network_type='drive'):
    """下载地图数据，带有重试机制"""
    print(f"Downloading street network for {place_name}...")
    return ox.graph_from_place(place_name, network_type=network_type)

def main():
    ########################################################################
    # 地图数据加载

    # 在线下载模式，需要VPN
    # 设置目标区域，例如城市名称，或指定经纬度边界框
    # place_name = "Haikou, Hainan, China"  # 替换目标区域
    # 下载街道网络数据（指定网络类型为 'drive' 仅包含可行驶的道路）
    # G = ox.graph_from_place(place_name, network_type='drive')
    # ox.save_graphml(G, filepath, gephi=False, encoding='utf-8')

    # # 离线加载模式
    # filepath = "input/map/haikou.graphml"  # 替换为目标文件路径
    # G = ox.load_graphml(filepath)
    # ox.plot_graph(G, ax=None, figsize=(8, 8), bgcolor='#111111',
    #                       node_color='w', node_size=15, node_alpha=None, node_edgecolor='none', node_zorder=1,
    #                       edge_color='#999999', edge_linewidth=1, edge_alpha=None,
    #                       bbox=None, show=True, close=False, save=False, filepath=None, dpi=300)
    # save_graph_shapefile_directional(G, filepath='input/map/haikou', encoding="utf-8")

    # Download by place name with retry logic

    try:
        filepath = "input/map/shanghai.graphml"
        G = ox.load_graphml(filepath)
        # place = "Shanghai, China"
        # G = download_graph_with_retry(place, network_type='drive')
        print("Graph downloaded successfully!")

        # Save the graph with progress tracking
        save_graph_shapefile_directional(G, filepath='input/map/shanghai', show_progress=True)
        print("Shapefiles saved successfully!")

    except Exception as e:
        print(f"Failed to download graph after multiple retries: {e}")
        print("Please check your internet connection and VPN settings.")
        print("Alternatively, you can use offline mode by loading from a .graphml file.")
        raise

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 打印异常堆栈信息
        traceback.print_exc()
        # 在出现异常的环境中进入pdb
        pdb.set_trace()