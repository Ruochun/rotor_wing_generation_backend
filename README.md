# Rotor Wing Generation Backend

This repository contains tools for generating parametric wing designs for rotors.

## Scripts

### `generate_params.py`

Generates a CSV file with wing design parameters from abstract design requirements.

**Usage:**

```bash
python generate_params.py input.csv [options]
```

**Options:**

- `--overall-length`: Overall length of the wing in meters (default: 0.02)
- `--chord-max-thickness`: Maximum chord thickness as percentage (default: 9.0)
- `--max-camber`: Maximum camber as percentage (default: 6.0)
- `--max-camber-location`: Location of max camber [0,1], where 0=leading edge (default: 0.4)
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
python generate_params.py input.csv

# Custom overall length and number of wings
python generate_params.py input.csv --overall-length 0.03 --n-wings 4

# Custom chord thickness, camber, and camber location
python generate_params.py input.csv --chord-max-thickness 12 --max-camber 5 --max-camber-location 0.3

# Custom chord distribution with high variance
python generate_params.py input.csv --average-chord-length 0.0025 --chord-length-variance 0.8

# Custom twist angle
python generate_params.py input.csv --max-twist-angle 45

# A full example
python generate_params.py input.csv --overall-length 0.025 --n-wings 2 --average-chord-length 0.004 --chord-length-variance 0 --max-twist-angle 10 --chord-max-thickness 25 --max-camber-location 0.4 --max-camber 6

python generate_params.py input.csv --overall-length 0.025 --n-wings 3 --average-chord-length 0.004 --chord-length-variance 0.6 --max-twist-angle 40 --chord-max-thickness 25 --max-camber-location 0.4 --max-camber 6
```

### `generate_wing.py`

Generates 3D wing geometry from CSV parameters and exports as STL file. **Automatically creates both counterclockwise (CCW) and clockwise (CW) versions** of the wing design.

**Usage:**

```bash
python generate_wing.py input.csv --output wing.stl [options]
```

**Options:**

- `--row`: Row index to use from CSV (default: 0)
- `--output`: Output STL file path (default: wing_output.stl)
- `--blend-sections`: Number of blend sections between defined stations (default: 6)
- `--profile-points`: Number of points per airfoil side (default: 50)
- `--envelope-offset`: Envelope offset as fraction of chord (default: 0.03). 
  Adds a small, thin envelope around all NACA sections by offsetting in the outward 
  normal direction. This removes sharp edges (especially at the trailing edge and 
  wing tip), making the wing more 3D printing friendly. Also controls the tip fillet 
  reduction: the final fillet section will be (1 - envelope-offset) times the original size.
- `--tip-fillet-sections`: Number of additional sections at the wing tip for filleting (default: 5).
  These sections progressively decrease in size toward the tip, creating a smooth rounded 
  tip edge. Higher values create smoother fillets. Set to 0 to disable tip filleting.

**3D Printing Enhancement:**

The `--envelope-offset` parameter adds a thin envelope around all airfoil sections, which:
- Removes sharp trailing edges that are difficult to print with **smooth circular rounding**
- Creates a rounded cap at the trailing edge by interpolating multiple offset directions
- Rounds the wing tip for better printability
- Eliminates thin features that might break during printing
- Default value of 0.03 (3% of chord) provides a good balance

The trailing edge rounding uses angular interpolation to create a smooth transition between 
upper and lower surface offsets, eliminating the sharp corners that would otherwise occur.

To disable the envelope (not recommended for 3D printing), set `--envelope-offset 0.0`.

**Tip Filleting:**

The `--tip-fillet-sections` parameter adds progressively smaller NACA sections at the wing tip:
- Creates a smooth rounded tip edge instead of a flat cap
- Each fillet section decreases in chord and thickness using a smooth power curve
- The fillet size reduction is controlled by `--envelope-offset`: if envelope-offset is 0.03, 
  the final fillet section will be 97% (1 - 0.03) of the original tip section size
- The fillet extends beyond the final defined section by (envelope-offset × tip_chord)
- Default value of 5 sections provides good smoothness
- Higher values (e.g., 7-10) create even smoother, more gradual fillets
- Set to 0 to disable tip filleting and use a flat cap instead

**Output:**

The script generates **two STL files**:
- `<filename>.stl` - Counterclockwise rotation version (standard convention)
- `<filename>_cw.stl` - Clockwise rotation version (mirrored across XY plane with corrected normals)

Both files maintain proper outward-pointing normals for correct 3D printing and visualization.

**Examples:**

```bash
# Generate wing from parameters with default settings (includes tip fillet)
# This creates two files: wing.stl (CCW) and wing_cw.stl (CW)
python generate_wing.py sample_params.csv --output wing.stl

# Generate wing with larger envelope (more aggressive fillet and envelope offset)
python generate_wing.py sample_params.csv --output wing.stl --envelope-offset 0.05

# Generate wing with more tip fillet sections for extra smooth tip
python generate_wing.py sample_params.csv --output wing.stl --tip-fillet-sections 8

# Generate wing without tip fillet (uses flat cap at tip)
python generate_wing.py sample_params.csv --output wing.stl --tip-fillet-sections 0

# Generate wing without envelope (sharp edges, not recommended for 3D printing)
python generate_wing.py sample_params.csv --output wing.stl --envelope-offset 0.0
```

### `analysis.py`

Analyzes wing designs using Blade Element Momentum Theory (BEMT) and computes aerodynamic performance characteristics including thrust, power, and torque.

**Usage:**

```bash
python analysis.py input.csv results.csv [options]
```

**Positional Arguments:**

- `input.csv`: CSV file with wing design parameters (generated by `generate_params.py`)
- `results.csv`: Output CSV file where analysis results will be saved

**Options:**

- `--rpm`: Rotation speed in RPM (default: 3000)
- `--rho`: Air density in kg/m³ (default: 1.225 at sea level)
- `--mu`: Dynamic viscosity in Pa·s (default: 1.81e-5)
- `--altitude`: Altitude in meters (overrides rho if specified, default: 0.0)
- `--elements`: Number of blade elements for analysis (default: 20)
- `--max-iter`: Maximum BEMT iterations (default: 100)
- `--quiet`: Suppress detailed output

**Output Units:**

The analysis outputs use micro-scale units for better readability of small rotor measurements:
- Thrust: μN (micronewtons, 10⁻⁶ N)
- Power: μW (microwatts, 10⁻⁶ W)
- Torque: μN·mm (micronewton-millimeters, 10⁻⁹ N·m)

**Examples:**

```bash
# Analyze wing designs with default RPM (3000)
python analysis.py sample_params.csv results.csv

# Analyze with custom RPM
python analysis.py sample_params.csv results.csv --rpm 4000

# Analyze at altitude (adjusts air density automatically)
python analysis.py sample_params.csv results.csv --altitude 1000

# Quiet mode (minimal output)
python analysis.py sample_params.csv results.csv --quiet
```

## Complete Workflow

Generate a custom wing design from abstract requirements and analyze its performance:

```bash
# Step 1: Generate parameter CSV from abstract requirements
python generate_params.py custom_params.csv \
    --overall-length 0.025 \
    --chord-max-thickness 12 \
    --max-camber 5 \
    --max-camber-location 0.3 \
    --average-chord-length 0.0022 \
    --chord-length-variance 0.7 \
    --max-twist-angle 45 \
    --n-wings 4

# Step 2: Generate 3D wing geometry (creates both CCW and CW versions)
# Default includes tip filleting with 5 sections for smooth rounded tips
python generate_wing.py custom_params.csv --output custom_wing.stl
# This creates: custom_wing.stl (CCW) and custom_wing_cw.stl (CW)

# Optional: Customize tip filleting and envelope offset
python generate_wing.py custom_params.csv --output custom_wing.stl --envelope-offset 0.05 --tip-fillet-sections 8

# Step 3: Analyze wing performance
python analysis.py custom_params.csv custom_analysis.csv --rpm 3000
```

## Parameter Translation

The `generate_params.py` script translates abstract design requirements into detailed parameters:

| Abstract Requirement | Translates To | Description |
|---------------------|---------------|-------------|
| `overall_length` | `overall_length` | Direct mapping |
| `chord_max_thickness` + `max_camber` + `max_camber_location` | 6x `naca_X` codes | 4-digit NACA airfoil codes |
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
- manifold3d (required for Boolean operations in hub generation)
- networkx (optional, for mesh repair operations)

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy trimesh manifold3d networkx
```

## Distribution as Executables

For users who don't want to install Python or dependencies, this project can be packaged into standalone executables for Windows and Linux.

### Using Pre-built Executables

If you received pre-built executables, you can use them directly without installing Python:

**Windows:**
```cmd
generate_params.exe input.csv --overall-length 0.03
generate_wing.exe input.csv --output wing.stl
analysis.exe input.csv results.csv --rpm 4000
```

**Linux:**
```bash
./generate_params input.csv --overall-length 0.03
./generate_wing input.csv --output wing.stl
./analysis input.csv results.csv --rpm 4000
```

### Building Executables

To build executables yourself, see the [packaging/README.md](packaging/README.md) for detailed instructions.

Quick start:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. Build executables:
   
   **Windows:**
   ```cmd
   cd packaging
   build_windows.bat
   ```
   
   **Linux/macOS:**
   ```bash
   cd packaging
   ./build_linux.sh
   ```

3. Find executables in `packaging/dist/`
