#!/usr/bin/env python3
"""Build ground_truth.csv for real_data from manually labeled GT edge segments."""
import csv, json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/real_data"

# ── Traj 11 GT segments: edge_id, start_seq, end_seq ──
# "0" = no road in edges.shp (gas station, lack roads, parked, under bridge)
TRAJ11_SEGMENTS = [
    # Sub-trajectory 1: seq 0–419 (end at road head reverse)
    (8088,   0,   105),
    (8088,   106, 191),
    (149059, 192, 199),
    (149060, 200, 214),
    (3585,   215, 231),
    (3583,   232, 246),
    (105747, 247, 254),
    (105749, 255, 284),
    (105750, 285, 306),
    (112798, 307, 317),
    (112801, 318, 323),
    (149100, 324, 336),
    (149098, 337, 354),
    (149099, 355, 358),
    (149099, 359, 366),   # no suitable road
    (149108, 367, 419),   # head reverse, same road

    # Sub-trajectory 2: seq 420–495 (110s gap at 419→420)
    (105740, 420, 425),
    (1890,   426, 495),

    # Sub-trajectory 3: seq 496–575 (gap at 495→496)
    (11158,  496, 575),

    # Sub-trajectory 4: seq 576–635
    (11158,  576, 582),   # no proper road
    (11158,  583, 635),   # reversing head

    # Sub-trajectory 5: seq 636–869
    (101366, 636, 656),   # no proper road, finish reverse
    (101366, 657, 869),

    # Sub-trajectory 6: seq 870–1539 (repeat of earlier route)
    (8088,   870,  1084),
    (149059, 1085, 1091),
    (149060, 1092, 1105),
    (3585,   1106, 1121),
    (3583,   1122, 1135),
    (105747, 1136, 1144),
    (105749, 1145, 1172),
    (105750, 1173, 1194),
    (112798, 1195, 1205),
    (112801, 1206, 1211),
    (149100, 1212, 1227),
    (149098, 1228, 1235),  # reversing head
    (149098, 1236, 1244),  # fully reversed
    (149099, 1245, 1249),
    (149109, 1250, 1267),
    (112799, 1268, 1281),
    (149102, 1282, 1306),
    (105741, 1307, 1325),
    (105740, 1326, 1345),
    (1890,   1346, 1539),

    # Sub-trajectory 7: seq 1540–1772
    (20104,  1540, 1546),
    (11156,  1547, 1599),  # RTK 1582-1583 repeat, deleted 1583
    (11157,  1600, 1618),  # reverse on un-modeled road
    (65439,  1619, 1696),  # reversing on lack roads
    (101366, 1697, 1726),  # reversing on lack roads
    (11152,  1727, 1747),  # reversing on lack roads
    (0,      1748, 1772),  # inner gas station, no road

    # Sub-trajectory 8: seq 1773–2695 (third pass of same route + end)
    (8088,   1773, 1978),
    (149059, 1979, 1984),
    (149060, 1985, 1999),
    (3585,   2000, 2017),
    (3583,   2018, 2033),
    (105747, 2034, 2042),
    (105749, 2043, 2071),
    (105750, 2072, 2094),
    (112798, 2095, 2108),
    (112801, 2109, 2115),
    (149100, 2116, 2128),
    (149098, 2129, 2159),
    (149099, 2160, 2164),
    (149109, 2165, 2181),
    (112799, 2182, 2197),
    (149102, 2198, 2219),
    (105741, 2220, 2235),
    (105740, 2236, 2252),
    (1890,   2253, 2432),
    (20103,  2433, 2446),  # reverse on lack road
    (11159,  2447, 2462),  # reverse on lack road
    (11156,  2463, 2501),
    (11157,  2502, 2527),  # lack road, use near one
    (65441,  2528, 2549),
    (0,      2550, 2588),  # reversing under bridge, no road
    (0,      2589, 2611),  # lack roads
    (11152,  2612, 2631),
    (0,      2632, 2658),  # gas station inner
    (0,      2659, 2695),  # parked
]


def main():
    # Load SPP observations to get timestamps per seq
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

    # Build per-epoch GT edge list for traj 11
    tid = 11
    max_seq = max(end for _, _, end in TRAJ11_SEGMENTS)
    point_edge_ids = [None] * (max_seq + 1)

    for edge_id, start, end in TRAJ11_SEGMENTS:
        for seq in range(start, end + 1):
            point_edge_ids[seq] = str(edge_id) if edge_id != 0 else "0"

    # Filter to only epochs with SPP observations, write point-format CSV
    out = DATA_DIR / "ground_truth.csv"
    total = 0
    unmatched = 0
    unique_edges = set()
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "seq", "timestamp", "x", "y", "edge_id"])
        for seq in range(len(point_edge_ids)):
            if (tid, seq) not in obs or point_edge_ids[seq] is None:
                continue
            eid = point_edge_ids[seq]
            o = obs[(tid, seq)]
            w.writerow([str(tid), str(seq), o["ts"], o["x"], o["y"], eid])
            total += 1
            if eid == "0":
                unmatched += 1
            else:
                unique_edges.add(eid)

    print(f"Saved: {out}")
    print(f"Traj {tid}: {total} GT epochs, {unmatched} no-road ({100*unmatched/total:.1f}%), "
          f"{len(unique_edges)} unique edges")
    print(f"Edge IDs: {sorted(unique_edges, key=int)}")

    spp_epochs = sum(1 for k in obs if k[0] == tid)
    print(f"SPP epochs: {spp_epochs}, GT covered: {total}, gap: {spp_epochs - total}")


if __name__ == "__main__":
    main()
