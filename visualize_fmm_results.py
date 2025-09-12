#!/usr/bin/env python3
"""
FMM地图匹配结果可视化脚本
将匹配结果叠加在道路网络上显示
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import numpy as np
from shapely.wkt import loads
import warnings
warnings.filterwarnings('ignore')

class FMMVisualizer:
    def __init__(self, data_dir="/home/dell/Czhang/fmm_sjtugnc/example/data"):
        self.data_dir = data_dir
        self.edges_file = f"{data_dir}/edges.shp"
        self.nodes_file = f"{data_dir}/nodes.shp"
        self.mr_file = "/home/dell/Czhang/fmm_sjtugnc/example/command_line_example/mr.txt"
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载道路网络和匹配结果数据"""
        print("正在加载数据...")
        
        # 加载道路网络
        self.edges_gdf = gpd.read_file(self.edges_file)
        self.nodes_gdf = gpd.read_file(self.nodes_file)
        
        print(f"道路网络: {len(self.edges_gdf)} 条边, {len(self.nodes_gdf)} 个节点")
        print(f"坐标范围: X({self.edges_gdf.total_bounds[0]:.2f} - {self.edges_gdf.total_bounds[2]:.2f}), "
              f"Y({self.edges_gdf.total_bounds[1]:.2f} - {self.edges_gdf.total_bounds[3]:.2f})")
        
        # 加载匹配结果
        self.mr_df = pd.read_csv(self.mr_file, sep=';')
        print(f"匹配结果: {len(self.mr_df)} 条轨迹")
        print(f"数据列: {list(self.mr_df.columns)}")
        
        # 解析匹配路径几何
        self.mr_df['matched_geometry'] = self.mr_df['mgeom'].apply(lambda x: loads(x) if pd.notna(x) else None)
    
    def create_interactive_map(self, output_file='fmm_results_map.html'):
        """创建交互式地图"""
        print(f"\n创建交互式地图...")
        
        # 计算地图中心
        bounds = self.edges_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # 创建地图
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # 添加道路网络
        print("  - 添加道路网络...")
        for idx, edge in self.edges_gdf.iterrows():
            if edge.geometry.geom_type == 'LineString':
                coords = list(edge.geometry.coords)
                folium.PolyLine(
                    locations=[[coord[1], coord[0]] for coord in coords],
                    color='gray',
                    weight=2,
                    opacity=0.6
                ).add_to(m)
        
        # 添加节点
        print("  - 添加节点...")
        for idx, node in self.nodes_gdf.iterrows():
            if node.geometry.geom_type == 'Point':
                coords = node.geometry.coords[0]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=3,
                    color='black',
                    fill=True,
                    fillColor='black',
                    fillOpacity=0.8
                ).add_to(m)
        
        # 添加匹配结果
        print("  - 添加匹配轨迹...")
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx, row in self.mr_df.iterrows():
            geom = row['matched_geometry']
            if geom and geom.geom_type == 'LineString':
                coords = list(geom.coords)
                color = colors[idx % len(colors)]
                
                # 匹配路径
                folium.PolyLine(
                    locations=[[coord[1], coord[0]] for coord in coords],
                    color=color,
                    weight=4,
                    opacity=0.9,
                    popup=f"""
                    <b>轨迹 ID: {row['id']}</b><br>
                    匹配路径: {row['opath']}<br>
                    累积概率: {row.get('cumu_prob', 'N/A')}<br>
                    路径长度: {sum([float(x) for x in row['spdist'].split(',')]):.3f}
                    """
                ).add_to(m)
                
                # 起点
                folium.Marker(
                    location=[coords[0][1], coords[0][0]],
                    icon=folium.Icon(color='green', icon='play'),
                    popup=f"起点 - 轨迹 {row['id']}"
                ).add_to(m)
                
                # 终点
                folium.Marker(
                    location=[coords[-1][1], coords[-1][0]],
                    icon=folium.Icon(color='red', icon='stop'),
                    popup=f"终点 - 轨迹 {row['id']}"
                ).add_to(m)
        
        # 保存地图
        m.save(output_file)
        print(f"交互式地图已保存: {output_file}")
        
        return m
    
    def create_static_plot(self, output_file='fmm_results_plot.png'):
        """创建静态地图"""
        print(f"\n创建静态地图...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 绘制道路网络
        print("  - 绘制道路网络...")
        self.edges_gdf.plot(ax=ax, color='lightgray', linewidth=1.5, alpha=0.7, label='道路网络')
        self.nodes_gdf.plot(ax=ax, color='black', markersize=20, alpha=0.8, label='节点')
        
        # 绘制匹配结果
        print("  - 绘制匹配轨迹...")
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        matched_geometries = []
        for idx, row in self.mr_df.iterrows():
            geom = row['matched_geometry']
            if geom:
                matched_geometries.append(geom)
                color = colors[idx % len(colors)]
                
                # 绘制匹配路径
                if geom.geom_type == 'LineString':
                    x_coords = [coord[0] for coord in geom.coords]
                    y_coords = [coord[1] for coord in geom.coords]
                    
                    ax.plot(x_coords, y_coords, color=color, linewidth=3, 
                           label=f'匹配轨迹 {row["id"]}', alpha=0.9)
                    
                    # 标记起点和终点
                    ax.plot(x_coords[0], y_coords[0], 'o', color='green', 
                           markersize=10, markeredgecolor='black', markeredgewidth=1)
                    ax.plot(x_coords[-1], y_coords[-1], 's', color='red', 
                           markersize=10, markeredgecolor='black', markeredgewidth=1)
                    
                    # 添加轨迹标签
                    mid_idx = len(geom.coords) // 2
                    mid_x, mid_y = geom.coords[mid_idx]
                    ax.annotate(f'T{row["id"]}', (mid_x, mid_y), 
                              xytext=(5, 5), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='white', alpha=0.8),
                              fontsize=10, fontweight='bold')
        
        # 设置图表属性
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title('FMM地图匹配结果可视化\n(匹配轨迹叠加在道路网络上)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 设置坐标轴范围
        bounds = self.edges_gdf.total_bounds
        margin = 0.2
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"静态地图已保存: {output_file}")
        
        plt.show()
        return fig
    
    def analyze_results(self):
        """分析匹配结果"""
        print("\n" + "="*50)
        print("FMM匹配结果分析")
        print("="*50)
        
        print(f"\n数据概览:")
        print(f"  道路网络: {len(self.edges_gdf)} 条边, {len(self.nodes_gdf)} 个节点")
        print(f"  匹配轨迹: {len(self.mr_df)} 条")
        
        print(f"\n匹配结果统计:")
        for col in ['id', 'opath', 'spdist', 'cpath', 'tpath']:
            if col in self.mr_df.columns:
                print(f"  {col}:")
                print(f"    示例值: {self.mr_df[col].iloc[0]}")
        
        print(f"\n路径长度分析:")
        total_lengths = []
        for idx, row in self.mr_df.iterrows():
            if 'spdist' in row:
                try:
                    lengths = [float(x) for x in row['spdist'].split(',')]
                    total_length = sum(lengths)
                    total_lengths.append(total_length)
                    print(f"    轨迹 {row['id']}: {total_length:.3f} (分段: {lengths})")
                except:
                    pass
        
        if total_lengths:
            print(f"    平均路径长度: {np.mean(total_lengths):.3f}")
            print(f"    总路径长度: {sum(total_lengths):.3f}")
        
        print(f"\n匹配质量指标:")
        if 'cumu_prob' in self.mr_df.columns:
            try:
                cumu_probs = [float(x.split(',')[-1]) if pd.notna(x) else 0 
                            for x in self.mr_df['cumu_prob']]
                print(f"    平均累积概率: {np.mean(cumu_probs):.3f}")
                print(f"    最小累积概率: {np.min(cumu_probs):.3f}")
                print(f"    最大累积概率: {np.max(cumu_probs):.3f}")
            except:
                print("    无法解析累积概率")
    
    def run_visualization(self):
        """运行完整的可视化流程"""
        print("开始FMM结果可视化...")
        
        # 分析结果
        self.analyze_results()
        
        # 创建可视化
        print(f"\n开始创建可视化...")
        
        # 静态地图
        self.create_static_plot()
        
        # 交互式地图
        self.create_interactive_map()
        
        print(f"\n✅ 可视化完成!")
        print(f"📊 静态地图: fmm_results_plot.png")
        print(f"🗺️  交互式地图: fmm_results_map.html")

def main():
    """主函数"""
    visualizer = FMMVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()