import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import osmnx as ox


CITY_CONFIG: Dict[str, Dict[str, object]] = {
    "haikou": {
        "query": "Haikou, Hainan, China",
        "label": "海口市 Haikou",
        "aliases": ["海口", "haikoushi", "haikou-shi"],
    },
    "sanya": {
        "query": "Sanya, Hainan, China",
        "label": "三亚市 Sanya",
        "aliases": ["三亚", "sanyashi"],
    },
    "sansha": {
        "query": "Sansha, Hainan, China",
        "label": "三沙市 Sansha",
        "aliases": ["三沙", "sanshashi"],
    },
    "danzhou": {
        "query": "Danzhou, Hainan, China",
        "label": "儋州市 Danzhou",
        "aliases": ["儋州", "danzhoushi"],
    },
    "wenchang": {
        "query": "Wenchang, Hainan, China",
        "label": "文昌市 Wenchang",
        "aliases": ["文昌", "wenchangshi"],
    },
    "qionghai": {
        "query": "Qionghai, Hainan, China",
        "label": "琼海市 Qionghai",
        "aliases": ["琼海", "qionghai", "qionghaishi"],
    },
    "wanning": {
        "query": "Wanning, Hainan, China",
        "label": "万宁市 Wanning",
        "aliases": ["万宁", "wanningshi"],
    },
    "dongfang": {
        "query": "Dongfang, Hainan, China",
        "label": "东方市 Dongfang",
        "aliases": ["东方", "dongfangshi"],
    },
    "wuzhishan": {
        "query": "Wuzhishan, Hainan, China",
        "label": "五指山市 Wuzhishan",
        "aliases": ["五指山", "wuzhishanshi"],
    },
    "lingshui": {
        "query": "Lingshui Li Autonomous County, Hainan, China",
        "label": "陵水黎族自治县 Lingshui",
        "aliases": ["陵水", "lingshui", "lingshui county"],
    },
    "baoting": {
        "query": "Baoting Li and Miao Autonomous County, Hainan, China",
        "label": "保亭黎族苗族自治县 Baoting",
        "aliases": ["保亭", "baoting"],
    },
    "baisha": {
        "query": "Baisha Li Autonomous County, Hainan, China",
        "label": "白沙黎族自治县 Baisha",
        "aliases": ["白沙", "baisha"],
    },
    "changjiang": {
        "query": "Changjiang Li Autonomous County, Hainan, China",
        "label": "昌江黎族自治县 Changjiang",
        "aliases": ["昌江", "changjiang"],
    },
    "chengmai": {
        "query": "Chengmai County, Hainan, China",
        "label": "澄迈县 Chengmai",
        "aliases": ["澄迈", "chengmai"],
    },
    "dingan": {
        "query": "Ding'an County, Hainan, China",
        "label": "定安县 Ding'an",
        "aliases": ["定安", "dingan"],
    },
    "lingao": {
        "query": "Lingao County, Hainan, China",
        "label": "临高县 Lingao",
        "aliases": ["临高", "lingao"],
    },
    "qiongzhong": {
        "query": "Qiongzhong Li and Miao Autonomous County, Hainan, China",
        "label": "琼中黎族苗族自治县 Qiongzhong",
        "aliases": ["琼中", "qiongzhong"],
    },
    "tunchang": {
        "query": "Tunchang County, Hainan, China",
        "label": "屯昌县 Tunchang",
        "aliases": ["屯昌", "tunchang"],
    },
    "ledong": {
        "query": "Ledong Li Autonomous County, Hainan, China",
        "label": "乐东黎族自治县 Ledong",
        "aliases": ["乐东", "ledong"],
    },
}


def normalize_token(value: str) -> str:
    return re.sub(r"[\s_\-]", "", value.strip().lower())


def build_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for city_key, info in CITY_CONFIG.items():
        alias_map[normalize_token(city_key)] = city_key
        alias_map[normalize_token(str(info.get("label", "")))] = city_key
        for alias in info.get("aliases", []):
            alias_map[normalize_token(alias)] = city_key
    return alias_map


ALIAS_MAP = build_alias_map()


def resolve_city_tokens(tokens: Sequence[str]) -> List[str]:
    resolved = []
    for token in tokens:
        key = ALIAS_MAP.get(normalize_token(token))
        if not key:
            raise ValueError(f"未识别的城市或县域标识: {token}")
        resolved.append(key)
    return resolved


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and manage Hainan drive road network data in chunks."
    )
    parser.add_argument("--list", action="store_true", help="列出支持的市县并退出。")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="仅下载指定的市县（名称或编号）。",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="跳过指定的市县（名称或编号）。",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="禁用交互选择，直接按参数执行。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="强制重新下载，即使已有缓存。",
    )
    parser.add_argument(
        "--skip-shapefile",
        action="store_true",
        help="只保存GraphML文件，跳过Shapefile导出。",
    )
    parser.add_argument(
        "--output-root",
        default="../input/map",
        help="输出根目录。",
    )
    parser.add_argument(
        "--province-dir",
        default="hainan",
        help="在输出根目录下的子目录名称。",
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        help="OSMnx下载的道路网络类型。",
    )
    parser.add_argument(
        "--retain-all",
        action="store_true",
        help="保留下载图中的所有连通分量。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="设置OSMnx请求超时时间（秒）。",
    )
    parser.add_argument(
        "--merge-output",
        help="指定拼接后的输出目录，自动合并选定市县的图数据。",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="只执行拼接，不触发下载流程。",
    )
    parser.add_argument(
        "--merge-extra",
        nargs="*",
        default=[],
        help="额外用于拼接的路径（GraphML文件或包含graph.graphml的目录）。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志等级，默认INFO，可设置为DEBUG。",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def list_cities() -> None:
    print("支持的市县：")
    for idx, (key, info) in enumerate(sorted(CITY_CONFIG.items()), 1):
        label = info.get("label", key)
        print(f"{idx:>2}. {key} ({label})")


def interactive_filter(cities: List[str]) -> List[str]:
    if not sys.stdin.isatty():
        return cities
    print("可下载的市县如下（输入编号或名称以排除，多项以空格或逗号分隔，直接回车表示全部下载）：")
    for idx, key in enumerate(cities, 1):
        label = CITY_CONFIG[key].get("label", key)
        print(f"{idx:>2}. {key} ({label})")
    raw = input("请输入需要排除的市县：").strip()
    if not raw:
        return cities
    tokens = re.split(r"[\s,;]+", raw)
    to_exclude = set()
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(cities):
                to_exclude.add(cities[idx])
            else:
                logging.warning("编号 %s 超出范围，将忽略。", token)
        else:
            try:
                resolved = resolve_city_tokens([token])[0]
                to_exclude.add(resolved)
            except ValueError as exc:
                logging.warning("%s", exc)
    return [city for city in cities if city not in to_exclude]


def build_city_selection(
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
    prompt: bool,
) -> List[str]:
    available = sorted(CITY_CONFIG.keys())
    if include:
        resolved = resolve_mixed_tokens(include, available)
        selected = [city for city in available if city in resolved]
    else:
        selected = available
    if exclude:
        resolved_exclude = resolve_mixed_tokens(exclude, available)
        selected = [city for city in selected if city not in resolved_exclude]
    if prompt:
        selected = interactive_filter(selected)
    if not selected:
        raise RuntimeError("未选择任何市县，任务已取消。")
    return selected


def resolve_mixed_tokens(
    tokens: Sequence[str], reference: Sequence[str]
) -> List[str]:
    result: List[str] = []
    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(reference):
                result.append(reference[idx])
            else:
                raise ValueError(f"编号 {token} 超出范围。")
        else:
            resolved = resolve_city_tokens([token])[0]
            result.append(resolved)
    return result


def download_city_graph(
    city_key: str,
    network_type: str,
    retain_all: bool,
) -> nx.MultiDiGraph:
    info = CITY_CONFIG[city_key]
    query = str(info["query"])
    logging.debug("开始下载 %s (%s)", city_key, query)
    graph = ox.graph_from_place(
        query,
        network_type=network_type,
        retain_all=retain_all,
    )
    logging.debug("完成下载 %s", city_key)
    return graph


def safe_save_graphml(graph: nx.MultiDiGraph, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".part")
    ox.save_graphml(graph, temp_path)
    temp_path.replace(target)


def safe_save_shapefile(graph: nx.MultiDiGraph, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target_dir.with_name(target_dir.name + "_tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    ox.save_graph_shapefile(graph, temp_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    temp_dir.rename(target_dir)


def city_output_paths(base_dir: Path, city_key: str) -> Tuple[Path, Path]:
    city_dir = base_dir / city_key
    graphml_path = city_dir / "graph.graphml"
    shapefile_dir = city_dir / "shapefile"
    return graphml_path, shapefile_dir


def download_cities(
    cities: Sequence[str],
    base_dir: Path,
    network_type: str,
    retain_all: bool,
    overwrite: bool,
    skip_shapefile: bool,
) -> List[Path]:
    successes: List[Path] = []
    total = len(cities)
    for index, city in enumerate(cities, start=1):
        graphml_path, shapefile_dir = city_output_paths(base_dir, city)
        already_downloaded = graphml_path.exists()
        shapefile_ready = shapefile_dir.exists() and any(shapefile_dir.iterdir())
        status = f"[{index}/{total}] {city}"
        if already_downloaded and not overwrite:
            resolved_path = graphml_path.resolve()
            if skip_shapefile or shapefile_ready:
                logging.info("%s 已存在，跳过下载。", status)
                successes.append(resolved_path)
                continue
            logging.info("%s 已有GraphML，补齐缺失的Shapefile。", status)
            try:
                cached_graph = ox.load_graphml(graphml_path)
                if not skip_shapefile:
                    safe_save_shapefile(cached_graph, shapefile_dir)
                successes.append(resolved_path)
                continue
            except Exception as exc:  # noqa: BLE001
                logging.warning("%s 读取缓存失败，将重新下载：%s", status, exc)
        logging.info("%s 开始下载...", status)
        try:
            graph = download_city_graph(city, network_type, retain_all)
            safe_save_graphml(graph, graphml_path)
            if not skip_shapefile:
                safe_save_shapefile(graph, shapefile_dir)
            logging.info("%s 下载完成。", status)
            successes.append(graphml_path.resolve())
        except Exception as exc:  # noqa: BLE001
            logging.exception("%s 下载失败：%s", status, exc)
    return successes


def resolve_graphml_path(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() != ".graphml":
            raise ValueError(f"文件 {path} 不是GraphML。")
        return path.resolve()
    if path.is_dir():
        candidate = path / "graph.graphml"
        if candidate.exists():
            return candidate.resolve()
        raise ValueError(f"目录 {path} 中未找到 graph.graphml。")
    raise ValueError(f"路径 {path} 不存在。")


def merge_graphs(
    graph_paths: Sequence[Path],
    output_dir: Path,
    skip_shapefile: bool,
) -> Optional[Path]:
    if not graph_paths:
        logging.warning("没有可用的GraphML文件，跳过拼接。")
        return None
    logging.info("准备拼接 %d 份图数据。", len(graph_paths))
    graphs: List[nx.MultiDiGraph] = []
    for path in graph_paths:
        logging.info("加载 %s", path)
        graphs.append(ox.load_graphml(path))
    merged = compose_graphs(graphs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_graphml = output_dir / "graph.graphml"
    safe_save_graphml(merged, output_graphml)
    if not skip_shapefile:
        safe_save_shapefile(merged, output_dir / "shapefile")
    logging.info("拼接结果已保存至 %s", output_dir)
    return output_graphml


def compose_graphs(graphs: Sequence[nx.MultiDiGraph]) -> nx.MultiDiGraph:
    composed = nx.MultiDiGraph()
    for graph in graphs:
        composed = nx.compose(composed, graph)
    return composed


def main() -> None:
    args = parse_arguments()
    configure_logging(args.log_level)

    if args.timeout:
        ox.settings.timeout = args.timeout

    if args.list:
        list_cities()
        return

    base_dir = Path(args.output_root) / args.province_dir
    prompt = not args.no_prompt

    try:
        selected_cities = build_city_selection(
            include=args.include,
            exclude=args.exclude,
            prompt=prompt,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("%s", exc)
        sys.exit(1)

    graphml_paths: List[Path] = []

    if not args.merge_only:
        downloaded = download_cities(
            selected_cities,
            base_dir=base_dir,
            network_type=args.network_type,
            retain_all=args.retain_all,
            overwrite=args.overwrite,
            skip_shapefile=args.skip_shapefile,
        )
        graphml_paths.extend(downloaded)
    else:
        logging.info("按要求跳过下载，仅执行拼接。")

    # For merge we also accept selected cities that are cached.
    for city in selected_cities:
        graph_path, _ = city_output_paths(base_dir, city)
        if graph_path.exists() and graph_path not in graphml_paths:
            graphml_paths.append(graph_path.resolve())

    for extra in args.merge_extra:
        try:
            extra_path = resolve_graphml_path(Path(extra).expanduser())
            if extra_path not in graphml_paths:
                graphml_paths.append(extra_path)
        except ValueError as exc:
            logging.error("%s", exc)

    if args.merge_output:
        merge_graphs(graphml_paths, Path(args.merge_output), args.skip_shapefile)
    else:
        logging.info("未指定拼接输出目录，跳过拼接。")


if __name__ == "__main__":
    main()
