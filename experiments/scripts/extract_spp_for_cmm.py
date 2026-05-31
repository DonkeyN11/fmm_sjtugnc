#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pyproj import Transformer
import glob
import os
import math
from datetime import datetime, timezone, date

def dm_to_dd(dm_str):
    if not dm_str: return 0.0
    try:
        val = float(dm_str)
        degrees = int(val / 100)
        minutes = val - degrees * 100
        return degrees + minutes / 60
    except:
        return 0.0

def parse_nmea_time(time_str):
    if not time_str: return None
    try:
        # Expected format: HHMMSS or HHMMSS.SS
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = float(time_str[4:])
        return hh, mm, ss
    except:
        return None

def parse_nmea_date(date_str):
    if not date_str: return None
    try:
        # Expected format: DDMMYY
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = int(date_str[4:6]) + 2000
        return date(year, month, day)
    except:
        return None

def ellipse_to_cov_m(smaj, smin, orient_deg):
    """
    Convert NMEA GST error ellipse to covariance matrix in meters (East, North).
    orient_deg is degrees from true North clockwise.
    """
    theta = math.radians(orient_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    vmaj = smaj * smaj
    vmin = smin * smin
    
    # Variance in North and East
    var_n = vmaj * (cos_t**2) + vmin * (sin_t**2)
    var_e = vmaj * (sin_t**2) + vmin * (cos_t**2)
    # Covariance East-North
    cov_en = (vmaj - vmin) * sin_t * cos_t
    
    return np.array([[var_e, cov_en], [cov_en, var_n]])

def get_jacobian_inv(lon, lat, transformer):
    """
    Compute inverse Jacobian J^-1 = d(lon, lat) / d(x, y)
    at given (lon, lat) for EPSG:4326 to EPSG:32649.
    """
    eps = 1e-6 # in degrees
    try:
        x0, y0 = transformer.transform(lon, lat)
        x1, y1 = transformer.transform(lon + eps, lat)
        x2, y2 = transformer.transform(lon, lat + eps)
        
        # J = [dx/dlon dx/dlat; dy/dlon dy/dlat]
        J = np.array([
            [(x1 - x0) / eps, (x2 - x0) / eps],
            [(y1 - y0) / eps, (y2 - y0) / eps]
        ])
        
        return np.linalg.inv(J)
    except:
        # Fallback to simple spherical approximation
        r_earth = 6378137.0
        dlon_dx = 180.0 / (math.pi * r_earth * math.cos(math.radians(lat)))
        dlat_dy = 180.0 / (math.pi * r_earth)
        return np.array([[dlon_dx, 0], [0, dlat_dy]])

def process_spp_file(file_path, transformer):
    print(f"Processing: {file_path}")
    
    # Get ID from path
    parts = file_path.split(os.sep)
    try:
        traj_id = parts[-3]
    except:
        traj_id = "unknown"

    data_records = {} # time_str -> data
    current_date = None
    
    try:
        with open(file_path, 'r', encoding='ascii', errors='ignore') as f:
            for line in f:
                if '$GN' not in line and '$GP' not in line: continue
                idx = line.find('$')
                if idx < 0: continue
                content = line[idx:].split('*')[0]
                fields = content.split(',')
                if len(fields) < 2: continue
                msg_type = fields[0][3:]
                
                if msg_type == 'ZDA' and len(fields) >= 5:
                    current_date = date(int(fields[4]), int(fields[3]), int(fields[2]))
                elif msg_type == 'RMC' and len(fields) >= 10:
                    d = parse_nmea_date(fields[9])
                    if d: current_date = d
                
                if current_date:
                    time_str = fields[1]
                    if not time_str: continue
                    if time_str not in data_records:
                        data_records[time_str] = {'date': current_date}
                    
                    if msg_type == 'GGA' and len(fields) >= 10:
                        data_records[time_str]['lat'] = dm_to_dd(fields[2]) if fields[3] == 'N' else -dm_to_dd(fields[2])
                        data_records[time_str]['lon'] = dm_to_dd(fields[4]) if fields[5] == 'E' else -dm_to_dd(fields[4])
                        try: data_records[time_str]['alt'] = float(fields[9])
                        except: pass
                    elif msg_type == 'RMC' and len(fields) >= 6:
                        if 'lat' not in data_records[time_str]:
                            data_records[time_str]['lat'] = dm_to_dd(fields[3]) if fields[4] == 'N' else -dm_to_dd(fields[3])
                            data_records[time_str]['lon'] = dm_to_dd(fields[5]) if fields[6] == 'E' else -dm_to_dd(fields[5])
                    elif msg_type == 'GLL' and len(fields) >= 5:
                        if 'lat' not in data_records[time_str]:
                            data_records[time_str]['lat'] = dm_to_dd(fields[1]) if fields[2] == 'N' else -dm_to_dd(fields[1])
                            data_records[time_str]['lon'] = dm_to_dd(fields[3]) if fields[4] == 'E' else -dm_to_dd(fields[3])
                    elif msg_type == 'GST' and len(fields) >= 9:
                        try:
                            data_records[time_str]['smaj'] = float(fields[3])
                            data_records[time_str]['smin'] = float(fields[4])
                            data_records[time_str]['orient'] = float(fields[5])
                            data_records[time_str]['alt_sd'] = float(fields[8])
                        except: pass
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    results = []
    for t_str in sorted(data_records.keys()):
        rec = data_records[t_str]
        if 'lat' not in rec or 'lon' not in rec or 'smaj' not in rec:
            continue
        
        d_obj = rec['date']
        t_parts = parse_nmea_time(t_str)
        if not t_parts: continue
        
        try:
            dt = datetime(d_obj.year, d_obj.month, d_obj.day,
                          t_parts[0], t_parts[1], int(t_parts[2]), 
                          int((t_parts[2]-int(t_parts[2]))*1e6),
                          tzinfo=timezone.utc)
            ts = dt.timestamp()
        except: continue
        
        lon, lat = rec['lon'], rec['lat']
        smaj, smin, orient = rec['smaj'], rec['smin'], rec['orient']
        alt_sd = rec.get('alt_sd', 1.0)
        
        # 1. Covariance in meters (Local North-East frame)
        cov_m = ellipse_to_cov_m(smaj, smin, orient)
        
        # 2. Jacobian transformation
        J_inv = get_jacobian_inv(lon, lat, transformer)
        
        # 3. Covariance in degrees
        cov_deg = J_inv @ cov_m @ J_inv.T
        
        sde = math.sqrt(max(0, cov_deg[0, 0]))
        sdn = math.sqrt(max(0, cov_deg[1, 1]))
        sden = cov_deg[0, 1]
        
        # 4. Protection Level
        tr = cov_deg[0, 0] + cov_deg[1, 1]
        det = cov_deg[0, 0] * cov_deg[1, 1] - cov_deg[0, 1]**2
        term = math.sqrt(max(0, tr**2 - 4*det))
        lambda_max = (tr + term) / 2.0
        smaj_deg = math.sqrt(max(0, lambda_max))
        protection_level = 6.0 * smaj_deg
        
        try:
            traj_id_int = int(traj_id.replace('.', ''))
        except:
            traj_id_int = traj_id
        
        results.append({
            'id': traj_id_int,
            'timestamp': ts,
            'x': lon,
            'y': lat,
            'sde': sde,
            'sdn': sdn,
            'sdu': alt_sd,
            'sdne': sden,
            'sdeu': 0.0,
            'sdun': 0.0,
            'protection_level': protection_level
        })
        
    return results

def main():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    all_results = []
    files = glob.glob("dataset-hainan-06/*/实时定位结果/spp_solution.txt")
    files.sort()
    
    if not files:
        print("No spp_solution.txt files found.")
        return

    for f in files:
        res = process_spp_file(f, transformer)
        if res:
            all_results.extend(res)
            print(f"  Extracted {len(res)} points.")
        
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = "dataset-hainan-06/cmm_input_points.csv"
        # Use semicolon as delimiter as expected by the C++ code
        df.to_csv(output_file, index=False, sep=';')
        print(f"Success! Saved total {len(df)} points to {output_file}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()
