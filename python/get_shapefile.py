#! conda fmm
# -*- coding: utf-8 -*-
"""
Hainan 'drive' road network downloader (by city blocks) with resume, caching, merge & progress.

Requirements:
  conda install osmnx geopandas shapely tqdm prompt-toolkit

Usage (常用用法):
  # 1) 列出并交互式选择要下载的市县（不需要的可取消选择）
  python hainan_osmnx_drive_downloader.py download --province "海南省, 中国" --outdir output/maps

  # 2) 命令行直接排除若干区域（逗号分隔），无交互
  python hainan_osmnx_drive_downloader.py download --exclude "三沙市,琼中黎族苗族自治县" --outdir output/maps

  # 3) 仅下载指定区域（白名单）
  python hainan_osmnx_drive_downloader.py download --include "海口市,三亚市,儋州市" --outdir output/maps

  # 4) 拼接已下载分块 + 额外外部目录(如历史下载 input/map/haikou)
  python hainan_osmnx_drive_downloader.py stitch --indir output/maps --extra "input/map/haikou,input/map/sanya" --out "output/merged/hainan_drive.shp"

Notes:
- 默认 network_type='drive'；每个分块会保存 nodes.shp 与 edges.shp（WGS84/EPSG:4326）。
- 使用 OSMnx 的请求缓存；中途中断后重跑会跳过已完成分块（断点续传）。
"""

import os
import re
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm

import osmnx as ox
# --- extra imports for robust networking ---
from requests.exceptions import SSLError, ConnectionError
from osmnx._errors import ResponseStatusCodeError

# 公开 Overpass 实例（写“/api”基址，OSMnx 会自动加 /interpreter）
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api",
    "https://overpass.osm.jp/api",
    "https://overpass-api.de/api",
    "https://overpass.openstreetmap.ru/api",
]

# Hainan 市/县清单（用于 Nominatim 回退路径）
HAINAN_AREAS = [
    "海口市", "三亚市", "儋州市", "三沙市",
    "五指山市", "文昌市", "琼海市", "万宁市", "东方市",
    "定安县", "屯昌县", "澄迈县", "临高县",
    "白沙黎族自治县", "昌江黎族自治县", "乐东黎族自治县",
    "陵水黎族自治县", "保亭黎族苗族自治县", "琼中黎族苗族自治县",
]

PROVINCE_NAME_DEFAULT = "海南省, 中国"

# ------------------------------ Utilities ------------------------------ #

def slugify(name: str) -> str:
    """文件夹友好化：中文名保留，去除空格和非法字符。"""
    s = name.strip()
    s = re.sub(r"[\/\\\:\*\?\"\<\>\|]", "_", s)
    s = re.sub(r"\s+", "", s)
    return s

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_manifest(outdir: Path, manifest: Dict):
    ensure_dir(outdir)
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def load_manifest(outdir: Path) -> Dict:
    f = outdir / "manifest.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}

def graph_to_shp(G, dst_dir: Path):
    """保存为 nodes/edges shapefile（WGS84）。"""
    ensure_dir(dst_dir)
    # 兼容不同 osmnx 版本的返回顺序（大多数为 nodes, edges）
    g1, g2 = ox.graph_to_gdfs(G)
    # 判定哪一个是 edges（有 'u','v' 字段的一般是 edges）
    if {"u", "v"}.issubset(set(g1.columns)):
        edges = g1
        nodes = g2
    else:
        nodes = g1
        edges = g2
    # 统一投影为 EPSG:4326
    if edges.crs is None:
        edges.set_crs(epsg=4326, inplace=True)
    else:
        edges = edges.to_crs(epsg=4326)
    if nodes.crs is None:
        nodes.set_crs(epsg=4326, inplace=True)
    else:
        nodes = nodes.to_crs(epsg=4326)
    nodes.to_file(dst_dir / "nodes.shp", driver="ESRI Shapefile", encoding="utf-8")
    edges.to_file(dst_dir / "edges.shp", driver="ESRI Shapefile", encoding="utf-8")

def has_complete_block(block_dir: Path) -> bool:
    """判断该分块是否已下载完成（edges.shp 存在即可）。"""
    return (block_dir / "edges.shp").exists()

# ------------------------- OSMnx Configuration ------------------------- #

def configure_osmnx(cache_dir: Path, timeout: int = 180, pause: float = 2.0):
    import osmnx as ox
    ensure_dir(cache_dir)

    # 缓存
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)

    # 正确设置网络参数（不要把 headers/timeout 放进 requests_kwargs）
    ox.settings.requests_timeout = int(timeout)
    ox.settings.requests_kwargs = {}  # 如需 proxies/verify/cert 可放这里，但不要含 timeout/headers

    # Nominatim 使用政策需要可识别 UA/Referer
    ox.settings.http_user_agent = "HainanDownloader/1.0 (contact: you@example.com)"
    ox.settings.http_referer = "https://example.com/hainan-downloader"
    ox.settings.http_accept_language = "zh-CN"

    # Overpass 设置模板（保持官方格式：不手搓分号）
    ox.settings.overpass_settings = "[out:json][timeout:{timeout}]{maxsize}"
    ox.settings.overpass_rate_limit = True  # 自动根据 /status 节点等待空闲槽
    ox.settings.overpass_url = OVERPASS_ENDPOINTS[0]  # 首选实例

    # 适度节流
    return float(pause)

def list_city_like_areas_via_nominatim(province_name: str = PROVINCE_NAME_DEFAULT) -> gpd.GeoDataFrame:
    """
    使用 Nominatim 逐个地名解析市/县边界，遵守 1 req/s。
    返回列：name, geometry
    """
    import osmnx as ox
    rows = []
    for name in HAINAN_AREAS:
        try:
            gdf = ox.geocode_to_gdf(f"{name}, {province_name}")
            if not gdf.empty:
                geom = gdf.geometry.iloc[0]
                rows.append({"name": name, "geometry": geom})
        except Exception as e:
            print(f"[nominatim] 跳过 {name}: {e}")
        time.sleep(1.05)  # 遵守 Nominatim 1 req/s
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ----------------------- Admin Boundaries Discovery -------------------- #

def get_province_polygon(province_name: str):
    gdf = ox.geocode_to_gdf(province_name)
    if gdf.empty:
        raise RuntimeError(f"无法定位省级名称：{province_name}")
    geom = gdf.geometry.iloc[0]
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    else:
        return geom.unary_union

def list_city_like_areas(province_poly, admin_levels=("6", "7"), province_name: str = PROVINCE_NAME_DEFAULT) -> gpd.GeoDataFrame:
    """
    优先用 Overpass 在省域内抓取 boundary=administrative 的市/县边界；
    如因超大范围或网络/SSL等失败，则回退到 Nominatim 逐名解析。
    返回列：name, admin_level(若可得), geometry
    """
    import osmnx as ox

    # 1) 尝试 Overpass：一次性获取省域全部行政边界
    try:
        tags = {"boundary": "administrative"}
        gdf = ox.features_from_polygon(province_poly, tags)
        if gdf.empty:
            raise RuntimeError("Overpass 返回空结果")

        cols = [c for c in ["name", "admin_level", "geometry"] if c in gdf.columns]
        gdf = gdf[cols].dropna(subset=["name"]).copy()
        if "admin_level" in gdf.columns:
            gdf = gdf[gdf["admin_level"].astype(str).isin(admin_levels)]

        # 同名 dissolve，保留最大面
        def largest_geom(geom):
            if isinstance(geom, Polygon):
                return geom
            if isinstance(geom, MultiPolygon):
                return max(list(geom.geoms), key=lambda a: a.area)
            return geom

        gdf = gdf.dissolve(by="name", as_index=False, aggfunc="first")
        gdf["geometry"] = gdf["geometry"].apply(largest_geom)
        gdf = gdf.sort_values(by="name").reset_index(drop=True)
        return gdf

    except (SSLError, ConnectionError, ResponseStatusCodeError, Exception) as e:
        print(f"[overpass] 行政边界拉取失败，回退 Nominatim：{e}")
        gdf2 = list_city_like_areas_via_nominatim(province_name=province_name)
        # Nominatim 不一定给 admin_level，这里补个占位列
        if not gdf2.empty and "admin_level" not in gdf2.columns:
            gdf2["admin_level"] = None
        return gdf2


# --------------------------- Interactive Picker ------------------------ #

def interactive_pick(to_download_names: List[str]) -> List[str]:
    """
    交互式勾选需要下载的区域（默认全选，空格切换，回车确认）。
    若未安装 prompt_toolkit 或终端不支持，直接返回原列表。
    """
    try:
        from prompt_toolkit.shortcuts import checkboxlist_dialog
        res = checkboxlist_dialog(
            title="选择需要下载的市/县（空格勾选，回车确定）",
            text="取消勾选即可跳过下载该区域：",
            values=[(name, name) for name in to_download_names],
        ).run()
        if res is None:
            return to_download_names  # 用户取消 -> 保持默认全选
        return list(res)
    except Exception:
        # 无交互环境时 fallback
        return to_download_names

# ----------------------------- Downloader ------------------------------ #

def download_one_area(name: str, geom, outdir: Path, network_type: str, pause: float,
                      max_retries: int = 4) -> Tuple[str, bool, Optional[str]]:
    import osmnx as ox

    slug = slugify(name)
    block_dir = outdir / slug
    if has_complete_block(block_dir):
        return name, True, None

    ensure_dir(block_dir)
    endpoint_idx = 0

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                # 指数退避
                time.sleep(pause * attempt)

            # 轮换 Overpass 实例（你前面已经定义了 OVERPASS_ENDPOINTS）
            ox.settings.overpass_url = OVERPASS_ENDPOINTS[endpoint_idx]

            # ★★★ 关键改动：去掉 clean_periphery，改用 truncate_by_edge=True（2.x 支持）
            G = ox.graph_from_polygon(
                geom,
                network_type=network_type,
                simplify=True,
                retain_all=True,
                truncate_by_edge=True,   # ← 2.x 建议使用的边界处理参数
                # custom_filter=None      # 如需自定义 Overpass 过滤器可在此传入
            )

            graph_to_shp(G, block_dir)
            (block_dir / "meta.json").write_text(
                json.dumps(
                    {"name": name, "slug": slug, "network_type": network_type,
                     "endpoint": ox.settings.overpass_url, "status": "ok"},
                    ensure_ascii=False, indent=2
                ),
                encoding="utf-8"
            )
            return name, True, None

        except (SSLError, ConnectionError, ResponseStatusCodeError) as e:
            endpoint_idx = (endpoint_idx + 1) % len(OVERPASS_ENDPOINTS)
            (block_dir / "error.log").write_text(
                f"attempt {attempt} @ {time.strftime('%F %T')}: {type(e).__name__}: {e}\n",
                encoding="utf-8"
            )
            continue

        except Exception as e:
            (block_dir / "error.log").write_text(
                f"attempt {attempt} @ {time.strftime('%F %T')}: {type(e).__name__}: {e}\n",
                encoding="utf-8"
            )
            if attempt >= max_retries:
                return name, False, str(e)

    return name, False, "max retries exceeded"



# ------------------------------- Stitcher ------------------------------ #

def discover_block_dirs(indir: Path) -> List[Path]:
    """发现 indir 下的分块目录（包含 edges.shp）。"""
    dirs = []
    for p in indir.iterdir():
        if p.is_dir() and (p / "edges.shp").exists():
            dirs.append(p)
    return sorted(dirs)

def load_edges(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def stitch_edges(indir: Path, extra_dirs: List[Path], out_path: Path):
    """拼接多个分块的 edges.shp（+ 额外目录），去重后输出单一 shapefile。"""
    all_dirs = discover_block_dirs(indir)
    for ed in extra_dirs:
        if ed.is_dir() and (ed / "edges.shp").exists():
            all_dirs.append(ed)
    if not all_dirs:
        raise RuntimeError("未发现可拼接的分块 edges.shp。")

    dfs = []
    for d in tqdm(all_dirs, desc="加载分块 edges"):
        try:
            gdf = load_edges(d / "edges.shp")
            gdf["__src__"] = d.name
            dfs.append(gdf)
        except Exception as e:
            print(f"跳过 {d}: {e}")

    merged = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs if dfs else "EPSG:4326")

    # 尝试按 osmid 与 几何去重（不同 OSMnx 版本字段略异，这里做尽量宽松的去重）
    osm_cols = [c for c in ["osmid", "u", "v", "key"] if c in merged.columns]
    merged["__geom_wkt__"] = merged.geometry.apply(lambda g: g.wkt if g is not None else "")
    dedup_cols = (osm_cols + ["__geom_wkt__"]) if osm_cols else ["__geom_wkt__"]
    merged = merged.drop_duplicates(subset=dedup_cols).drop(columns="__geom_wkt__", errors="ignore")

    ensure_dir(Path(out_path).parent)
    merged.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"✅ 拼接完成：{out_path}  （要素数：{len(merged)}）")

# ------------------------------- CLI Main ------------------------------ #

def cmd_download(args):
    pause = configure_osmnx(Path(args.cache), timeout=args.timeout, pause=args.pause)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # 省域及其市级/县级边界
    prov_poly = get_province_polygon(args.province)
    city_gdf = list_city_like_areas(prov_poly, admin_levels=tuple(args.admin_levels.split(",")))

    # include / exclude 处理
    names_all = list(city_gdf["name"])
    if args.include:
        include_set = {n.strip() for n in args.include.split(",") if n.strip()}
        names_pick = [n for n in names_all if n in include_set]
    else:
        names_pick = names_all.copy()

    if args.exclude:
        exclude_set = {n.strip() for n in args.exclude.split(",") if n.strip()}
        names_pick = [n for n in names_pick if n not in exclude_set]

    # 交互式勾选（仅在未显式指定 include 时启用）
    if args.interactive and not args.include:
        names_pick = interactive_pick(names_pick)

    if not names_pick:
        print("未选择任何区域，退出。")
        return

    # 清单
    manifest = {
        "province": args.province,
        "network_type": args.network_type,
        "admin_levels": args.admin_levels,
        "selected": names_pick,
        "outdir": str(outdir),
    }
    save_manifest(outdir, manifest)

    # 下载循环（带进度）
    city_map = {row["name"]: row.geometry for _, row in city_gdf.iterrows()}
    pbar = tqdm(total=len(names_pick), desc=f"下载（{args.network_type}）", ncols=100)
    ok, fail = [], []

    for name in names_pick:
        geom = city_map.get(name)
        if geom is None:
            fail.append((name, "无几何"))
            pbar.update(1)
            continue

        # 已完成直接跳过
        if has_complete_block(outdir / slugify(name)):
            ok.append(name)
            pbar.set_postfix_str(f"已存在：{name}")
            pbar.update(1)
            continue

        pbar.set_postfix_str(f"获取：{name}")
        nm, success, err = download_one_area(
            name=name,
            geom=geom,
            outdir=outdir,
            network_type=args.network_type,
            pause=pause,
            max_retries=args.retries,
        )
        if success:
            ok.append(nm)
            pbar.set_postfix_str(f"完成：{nm}")
        else:
            fail.append((nm, err))
            pbar.set_postfix_str(f"失败：{nm}")
        pbar.update(1)

        # 轻微节流，避免对 Overpass 压力（可调小/为0）
        time.sleep(args.pause)

    pbar.close()

    # 总结
    print("\n====== 总结 ======")
    print(f"✅ 成功 {len(ok)} 个：{', '.join(ok) if ok else '-'}")
    if fail:
        print(f"❌ 失败 {len(fail)} 个：")
        for n, e in fail:
            print(f"  - {n}: {e}")
    else:
        print("❌ 失败 0 个")

def cmd_stitch(args):
    indir = Path(args.indir)
    extra_dirs = [Path(p.strip()) for p in (args.extra.split(",") if args.extra else []) if p.strip()]
    out_path = Path(args.out)
    stitch_edges(indir, extra_dirs, out_path)

def build_parser():
    p = argparse.ArgumentParser(description="Hainan OSMnx 'drive' downloader (by city blocks) with resume/cache/stitch/progress")
    sub = p.add_subparsers(dest="cmd", required=True)

    # download subcommand
    d = sub.add_parser("download", help="按市级分块下载 'drive' 路网（支持断点续传与交互选择）")
    d.add_argument("--province", default="海南省, 中国", help="省级名称 (默认: 海南省, 中国)")
    d.add_argument("--admin-levels", default="6,7", help="行政层级（默认 6,7）")
    d.add_argument("--network-type", default="drive", help="OSMnx network_type (默认 drive)")
    d.add_argument("--outdir", required=True, help="输出目录（每个分块一个子文件夹，含 nodes/edges.shp）")
    d.add_argument("--cache", default="cache/osmnx", help="OSMnx 请求缓存目录")
    d.add_argument("--timeout", type=int, default=180, help="Overpass 请求超时秒数（默认 180）")
    d.add_argument("--pause", type=float, default=2.0, help="相邻请求的节流/等待秒数（默认 2.0）")
    d.add_argument("--retries", type=int, default=4, help="每个分块失败重试次数（默认 4）")
    d.add_argument("--include", default="", help="仅下载这些区域（逗号分隔）；留空为全部")
    d.add_argument("--exclude", default="", help="排除这些区域（逗号分隔）")
    d.add_argument("--interactive", action="store_true", help="交互式勾选要下载的区域（若指定 --include 则忽略）")
    d.set_defaults(func=cmd_download)

    # stitch subcommand
    s = sub.add_parser("stitch", help="拼接多个分块的 edges.shp（可加入历史目录）")
    s.add_argument("--indir", required=True, help="分块目录根（子目录下含 edges.shp）")
    s.add_argument("--extra", default="", help="额外历史分块目录（逗号分隔），如 input/map/haikou")
    s.add_argument("--out", required=True, help="输出的拼接 shapefile 路径，如 output/merged/hainan_drive.shp")
    s.set_defaults(func=cmd_stitch)

    return p

if __name__ == "__main__":
    # 延迟导入 pandas，避免 Cold start 慢
    import pandas as pd  # noqa: E402

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
