import pandas as pd
import plotly.graph_objects as go
import re
import os  # 导入 os 模块

# --- 配置 ---
# 1. 从环境变量中读取 Mapbox Access Token
#    脚本会自动寻找名为 MAPBOX_ACCESS_TOKEN 的环境变量
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

# 2. 指定包含经纬度坐标的CSV文件路径
CSV_FILE_PATH = "dataset_hainan_06/1.1/mr/cmm_trajectory.csv"

# --- 数据读取与解析 ---
def plot_trajectory_from_csv(csv_path: str, token: str):
    """
    从 cmm_trajectory.csv 文件中读取轨迹数据并在 Mapbox 上可视化。
    """
    # 检查是否成功从环境变量中获取了Token
    if not token:
        print("错误：未能从环境变量 'MAPBOX_ACCESS_TOKEN' 中读取到 Mapbox Access Token。")
        print("请确认您已在 .bashrc 中设置 'export MAPBOX_ACCESS_TOKEN=\"YOUR_TOKEN\"' 并且已经重新加载了 shell。")
        return

    try:
        # 使用 pandas 读取 CSV 文件，分隔符是分号
        df = pd.read_csv(csv_path, delimiter=';')
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        return

    # 提取 WKT (Well-Known Text) 格式的几何字符串
    wkt_string = df['geom'].iloc[0]

    # 使用正则表达式从 LINESTRING 中提取所有坐标对
    coords = re.findall(r"(-?\d+\.\d+)\s(-?\d+\.\d+)", wkt_string)

    if not coords:
        print("错误：未能在文件中找到有效的坐标。")
        return

    # 将坐标对拆分为经度(lon)和纬度(lat)列表
    longitudes = [float(lon) for lon, lat in coords]
    latitudes = [float(lat) for lon, lat in coords]

    print(f"成功解析 {len(longitudes)} 个坐标点。")

    # --- 创建地图 ---
    fig = go.Figure(go.Scattermapbox(
        mode="lines+markers",
        lon=longitudes,
        lat=latitudes,
        marker={'size': 5, 'color': 'blue'},
        line={'width': 2, 'color': 'blue'},
        name='Trajectory'
    ))

    # 更新地图布局
    fig.update_layout(
        title=f"Trajectory from {csv_path}",
        mapbox_style="mapbox://styles/mapbox/satellite-streets-v12",
        mapbox_accesstoken=token,
        mapbox={
            'center': {'lon': sum(longitudes) / len(longitudes), 'lat': sum(latitudes) / len(latitudes)},
            'zoom': 15
        },
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    # 显示地图
    fig.show()


if __name__ == "__main__":
    # 确保在修改 .bashrc 后，你已经执行了 `source ~/.bashrc` 或打开了一个新的终端
    plot_trajectory_from_csv(CSV_FILE_PATH, MAPBOX_ACCESS_TOKEN)
