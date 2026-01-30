# CMM Input Data Format Specification

This document describes the input data format required by the Covariance-based Map Matching (CMM) algorithm.

## Overview

CMM requires GPS trajectory data with additional covariance and protection level information for each point. The data can be provided in two CSV formats:

1. **Aggregated format** (recommended): Each row contains one complete trajectory
2. **Point-based format**: Each row contains one GPS point

## Aggregated CSV Format (Recommended)

### File Structure

The CSV file should use semicolon (`;`) as delimiter with the following columns:

| Column Name | Type | Description |
|-------------|------|-------------|
| `id` | integer | Unique trajectory identifier |
| `geom` | WKT | LINESTRING geometry in WKT format |
| `timestamps` | JSON array | Unix timestamps for each point |
| `covariances` | JSON 2D array | Covariance matrices for each point |
| `protection_levels` | JSON array | Protection level for each point |

### Format Details

#### 1. Geometry Column (`geom`)

Contains the trajectory geometry in Well-Known Text (WKT) format:

```
LINESTRING (x1 y1, x2 y2, x3 y3, ...)
```

Example:
```
LINESTRING (441797.80 2209756.27, 441797.72 2209756.17, 441797.75 2209756.29, ...)
```

#### 2. Timestamps Column (`timestamps`)

JSON 1D array containing Unix timestamps for each point:

```
[t1, t2, t3, ...]
```

Example:
```
[1234567890.0, 1234567891.0, 1234567892.0, ...]
```

#### 3. Covariances Column (`covariances`)

JSON 2D array where each inner array contains 6 values representing the covariance matrix for one point:

```
[[sde1, sdn1, sdu1, sdne1, sdeu1, sdun1], [sde2, sdn2, sdu2, sdne2, sdeu2, sdun2], ...]
```

**Covariance Matrix Components:**

- `sde`: East standard deviation (σ_e)
- `sdn`: North standard deviation (σ_n)
- `sdu`: Up standard deviation (σ_u)
- `sdne`: North-East covariance (σ_ne)
- `sdeu`: East-Up covariance (σ_eu)
- `sdun`: Up-North covariance (σ_un)

**Matrix Representation:**

```
| sde²   sdne   sdeu |
| sdne   sdn²   sdun |
| sdeu   sdun   sdu² |
```

Example:
```
[[0.679,0.691,0.809,0.033,0.0,0.0],[0.679,0.692,0.808,0.033,0.0,0.0],...]
```

#### 4. Protection Levels Column (`protection_levels`)

JSON 1D array containing protection level values for each point:

```
[pl1, pl2, pl3, ...]
```

Example:
```
[1.38, 1.38, 1.39, 1.37, ...]
```

### Complete Example

```csv
id;geom;timestamps;covariances;protection_levels
11;"LINESTRING (441797.80 2209756.27, 441797.72 2209756.17, 441797.75 2209756.29)";"[1234567890.0, 1234567891.0, 1234567892.0]";"[[0.679,0.691,0.809,0.033,0.0,0.0],[0.679,0.692,0.808,0.033,0.0,0.0],[0.678,0.693,0.811,0.032,0.0,0.0]]";"[1.38,1.38,1.39]"
```

## Point-Based CSV Format

Each row represents a single GPS point with the following columns:

| Column Name | Type | Description |
|-------------|------|-------------|
| `id` | integer | Trajectory identifier (multiple rows share the same id) |
| `timestamp` | double | Unix timestamp |
| `x` | double | X coordinate (e.g., longitude or projected X) |
| `y` | double | Y coordinate (e.g., latitude or projected Y) |
| `sdn` | double | North standard deviation |
| `sde` | double | East standard deviation |
| `sdu` | double | Up standard deviation |
| `sdne` | double | North-East covariance |
| `sdeu` | double | East-Up covariance |
| `sdun` | double | Up-North covariance |
| `protection_level` | double | Protection level value |

**Note:** The point-based format does not use JSON encoding. Each value is a plain number.

## Configuration

### XML Configuration

```xml
<config>
  <input>
    <gps>
      <file>your_data.csv</file>
      <id>id</id>
      <geom>geom</geom>
      <timestamp>timestamps</timestamp>
      <covariance>covariances</covariance>
      <protection_level>protection_levels</protection_level>
    </gps>
  </input>

  <parameters>
    <k>16</k>
    <min_candidates>1</min_candidates>
    <protection_level_multiplier>10.0</protection_level_multiplier>
    <filtered>true</filtered>
    <!-- other parameters... -->
  </parameters>
</config>
```

### Column Mapping

The column names in the configuration file should match the actual column headers in your CSV file. The parser is case-insensitive and supports common variations:

- `id` can be: `id`, `ID`, `Id`, etc.
- `covariance` can be: `covariances`, `covariance`, `covariance_json`, etc.
- `protection_level` can be: `protection_levels`, `protection_level`, `pl`, etc.

## Coordinate System

- Input coordinates should be in a consistent coordinate system
- For geographic coordinates (longitude/latitude), set `convert_to_projected=true` in the configuration
- The network CRS should match or be transformable from the input CRS

## Data Validation

CMM will validate that:
1. Number of points in geometry matches the length of timestamps array
2. Number of points matches the number of covariance matrices
3. Number of points matches the number of protection levels
4. Each covariance matrix has exactly 6 values
5. All numeric values can be parsed correctly

Invalid rows will be skipped with warning messages.

## Reference Implementation

See `src/mm/cmm/cmm_algorithm.cpp` for the parsing implementation:
- `parse_numeric_array()`: Parses timestamps and protection_levels
- `parse_covariance_array()`: Parses covariance matrices
