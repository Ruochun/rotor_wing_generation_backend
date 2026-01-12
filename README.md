# Rotor Wing Generation Backend

This repository contains tools for generating parametric wing designs for rotors.

## Scripts

### `generate_params.py`

Generates a CSV file with wing design parameters from abstract design requirements.

**Usage:**

```bash
python generate_params.py output.csv [options]
```

**Options:**

- `--overall-length`: Overall length of the wing in meters (default: 0.02)
- `--chord-max-thickness`: Maximum chord thickness as percentage (default: 9.0)
- `--chord-max-thickness-location`: Location of max thickness [0,1], where 0=leading edge (default: 0.4)
- `--average-chord-length`: Average chord length in meters (default: 0.002). Note: Root section is always fixed at 0.003m for rotor hub union.
- `--chord-length-variance`: Chord length variance [0,1], 0=constant, 1=max variation (default: 0.5). Uses smooth cosine-based transitions.
- `--max-twist-angle`: Maximum twist angle at root in degrees (default: 40.0)
- `--n-wings`: Number of wings (default: 3)
- `--rpm`: Rotations per minute (default: 3000.0)
- `--rho`: Air density in kg/m³ (default: 1.225)
- `--n-sections`: Number of sections along the wing (default: 6)

**Examples:**

```bash
# Generate with all defaults (similar to sample_params.csv)
python generate_params.py output.csv

# Custom overall length and number of wings
python generate_params.py output.csv --overall-length 0.03 --n-wings 4

# Custom chord thickness and location
python generate_params.py output.csv --chord-max-thickness 12 --chord-max-thickness-location 0.3

# Custom chord distribution with high variance
python generate_params.py output.csv --average-chord-length 0.0025 --chord-length-variance 0.8

# Custom twist angle
python generate_params.py output.csv --max-twist-angle 45

# A full example
python generate_params.py output.csv --overall-length 0.03 --n-wings 2 --average-chord-length 0.005 --chord-length-variance 0 --max-twist-angle 10 --chord-max-thickness 20
```

### `generate_wing.py`

Generates 3D wing geometry from CSV parameters and exports as OBJ file.

**Usage:**

```bash
python generate_wing.py input.csv --output wing.obj [options]
```

**Options:**

- `--row`: Row index to use from CSV (default: 0)
- `--output`: Output OBJ file path (default: wing_output.obj)
- `--blend-sections`: Number of blend sections between defined stations (default: 6)
- `--profile-points`: Number of points per airfoil side (default: 50)

**Example:**

```bash
# Generate wing from parameters
python generate_wing.py sample_params.csv --output wing.obj
```

## Complete Workflow

Generate a custom wing design from abstract requirements:

```bash
# Step 1: Generate parameter CSV from abstract requirements
python generate_params.py custom_params.csv \
    --overall-length 0.025 \
    --chord-max-thickness 12 \
    --chord-max-thickness-location 0.3 \
    --average-chord-length 0.0022 \
    --chord-length-variance 0.7 \
    --max-twist-angle 45 \
    --n-wings 4

# Step 2: Generate 3D wing geometry
python generate_wing.py custom_params.csv --output custom_wing.obj
```

## Parameter Translation

The `generate_params.py` script translates abstract design requirements into detailed parameters:

| Abstract Requirement | Translates To | Description |
|---------------------|---------------|-------------|
| `overall_length` | `overall_length` | Direct mapping |
| `chord_max_thickness` + `chord_max_thickness_location` | 6x `naca_X` codes | 4-digit NACA airfoil codes |
| `average_chord_length` + `chord_length_variance` | 6x `chord_X` lengths | Smooth chord distribution along span. Root (chord_0) is always 0.003m. |
| `max_twist_angle` | 6x `twist_X` angles | Linear interpolation from max to 0 |
| `n_wings` | `n_wings` | Direct mapping |

## Wing Generation

The `generate_wing.py` script uses **cubic spline interpolation** for smooth lofting between sections. The specified chord lengths and twist angles are treated as control points, and the actual wing surface is generated with smooth transitions using scipy's interpolation. This eliminates kinks and creates smooth leading and trailing edges.

### Multi-Wing Alignment

When generating multiple wings (n_wings > 1), the script automatically shifts each wing by 1/4 of the root chord length in the +X direction before rotation. This ensures that all wings align properly at their mid-chord (1/2 chord) position when rotated around the Y-axis, creating a symmetric multi-blade design.

## Requirements

- Python 3.7+
- numpy
- scipy
- trimesh

Install dependencies:

```bash
pip install numpy scipy trimesh
```
