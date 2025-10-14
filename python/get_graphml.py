import os
import osmnx as ox
import pdb
import traceback

def main():
    ########################################################################
    # 地图数据加载
    shapefile_dir = "../data/shapefile/haikou"  # 替换为目标输出目录
    # 在线下载模式，需要VPN
    # 设置目标区域，例如城市名称，或指定经纬度边界框
    place_name = "Hainan, China"  # 替换目标区域
    # 下载街道网络数据（指定网络类型为 'drive' 仅包含可行驶的道路）
    G = ox.graph_from_place(place_name, network_type='drive')
    # 创建保存目录
    os.makedirs(shapefile_dir, exist_ok=True)
    ox.save_graph_shapefile(G, filepath=shapefile_dir)
    
    # 离线加载模式
    
    # G = ox.load_graphml(filepath)
    # ox.plot_graph(G, ax=None, figsize=(8, 8), bgcolor='#111111', 
    #                       node_color='w', node_size=15, node_alpha=None, node_edgecolor='none', node_zorder=1, 
    #                       edge_color='#999999', edge_linewidth=1, edge_alpha=None, 
    #                       bbox=None, show=True, close=False, save=False, filepath=None, dpi=300)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 打印异常堆栈信息
        traceback.print_exc()
        # 在出现异常的环境中进入pdb
        pdb.set_trace()
