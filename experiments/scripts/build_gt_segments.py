#!/usr/bin/env python3
"""Build ground_truth.csv for real_data from manually labeled GT edge segments.

Usage:
  python experiments/scripts/build_gt_segments.py
"""

import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/real_data"

# ── GT segments: (edge_id, start_seq, end_seq) per trajectory ──
# edge_id=0 means no road in edges.shp

TRAJ11 = [
    (8088,   0,   105), (8088,   106, 191), (149059, 192, 199), (149060, 200, 214),
    (3585,   215, 231), (3583,   232, 246), (105747, 247, 254), (105749, 255, 284),
    (105750, 285, 306), (112798, 307, 317), (112801, 318, 323), (149100, 324, 336),
    (149098, 337, 354), (149099, 355, 358), (149099, 359, 366), (149108, 367, 419),
    (105740, 420, 425), (1890,   426, 495), (11158,  496, 575), (11158,  576, 582),
    (11158,  583, 635), (101366, 636, 656), (101366, 657, 869), (8088,   870, 1084),
    (149059, 1085,1091), (149060, 1092,1105), (3585,   1106,1121), (3583,   1122,1135),
    (105747, 1136,1144), (105749, 1145,1172), (105750, 1173,1194), (112798, 1195,1205),
    (112801, 1206,1211), (149100, 1212,1227), (149098, 1228,1235), (149098, 1236,1244),
    (149099, 1245,1249), (149109, 1250,1267), (112799, 1268,1281), (149102, 1282,1306),
    (105741, 1307,1325), (105740, 1326,1345), (1890,   1346,1539), (20104,  1540,1546),
    (11156,  1547,1599), (11157,  1600,1618), (65439,  1619,1696), (101366, 1697,1726),
    (11152,  1727,1747), (0,      1748,1772), (8088,   1773,1978), (149059, 1979,1984),
    (149060, 1985,1999), (3585,   2000,2017), (3583,   2018,2033), (105747, 2034,2042),
    (105749, 2043,2071), (105750, 2072,2094), (112798, 2095,2108), (112801, 2109,2115),
    (149100, 2116,2128), (149098, 2129,2159), (149099, 2160,2164), (149109, 2165,2181),
    (112799, 2182,2197), (149102, 2198,2219), (105741, 2220,2235), (105740, 2236,2252),
    (1890,   2253,2432), (20103,  2433,2446), (11159,  2447,2462), (11156,  2463,2501),
    (11157,  2502,2527), (65441,  2528,2549), (0,      2550,2588), (0,      2589,2611),
    (11152,  2612,2631), (0,      2632,2658), (0,      2659,2695),
]

TRAJ12 = [
    (0,     0,   349), (8088, 350, 400), (0,   401, 430), (8088, 431, 445),
    (0,   446, 474), (8088, 475, 490), (0,   491, 519), (8088, 520, 532),
    (0,   533, 554), (8088, 555, 573), (0,   574, 603), (8088, 604, 622),
    (0,   623, 733),
]

TRAJ13 = [
    # TODO: type edge ranges for traj 13 (id=13, 2391 obs epochs)
]

TRAJ14 = [
    # TODO: type edge ranges for traj 14 (id=14, 1370 obs epochs)
]

TRAJ21 = [
    # TODO: type edge ranges for traj 21 (id=21, 3565 obs epochs)
]

TRAJ22 = [
    # TODO: type edge ranges for traj 22 (id=22, 2490 obs epochs)
]

TRAJ23 = [
    # TODO: type edge ranges for traj 23 (id=23, 2721 obs epochs)
]

ALL_SEGMENTS = {11: TRAJ11, 12: TRAJ12, 13: TRAJ13, 14: TRAJ14,
                21: TRAJ21, 22: TRAJ22, 23: TRAJ23}


def main():
    # Load SPP observations for timestamps
    obs = {}
    with open(DATA_DIR / "cmm_input_points.csv", newline="") as f:
        seq_counters = defaultdict(int)
        for row in csv.DictReader(f, delimiter=";"):
            tid = int(row["id"])
            seq = seq_counters[tid]
            seq_counters[tid] += 1
            obs[(tid, seq)] = {
                "ts": row["timestamp"],
                "x": row["x"], "y": row["y"],
            }

    # Build GT per trajectory
    all_rows = []
    for tid, segments in sorted(ALL_SEGMENTS.items()):
        if not segments:
            continue

        max_seq = max(end for _, _, end in segments)
        point_edges = [None] * (max_seq + 1)
        for edge_id, start, end in segments:
            for seq in range(start, end + 1):
                point_edges[seq] = str(edge_id) if edge_id != 0 else "0"

        total = unmatched = 0
        unique_edges = set()
        for seq in range(len(point_edges)):
            if (tid, seq) not in obs or point_edges[seq] is None:
                continue
            eid = point_edges[seq]
            o = obs[(tid, seq)]
            all_rows.append({
                "id": str(tid), "seq": str(seq),
                "timestamp": o["ts"], "x": o["x"], "y": o["y"],
                "edge_id": eid,
            })
            total += 1
            if eid == "0":
                unmatched += 1
            else:
                unique_edges.add(eid)

        spp_epochs = sum(1 for k in obs if k[0] == tid)
        print(f"Traj {tid}: {total} GT epochs ({unmatched} no-road, {len(unique_edges)} unique edges), "
              f"SPP epochs: {spp_epochs}")

    # Write
    out = DATA_DIR / "ground_truth.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "seq", "timestamp", "x", "y", "edge_id"],
                           delimiter=";")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
