#!/usr/bin/env python3
"""
B-Rep-Based Wing Generation System

This module generates parametric wing designs from CSV parameters using B-Rep CAD operations.
It creates NURBS surfaces, performs boolean operations, and exports both B-Rep (STEP) and mesh (STL) formats.

The system uses build123d (a Python wrapper around OpenCascade) to create true CAD models
that can be further edited in CAD software, unlike the mesh-only approach in generate_wing.py.
"""

import argparse
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy import interpolate
from build123d import *
from build123d import export_step, export_stl, Unit

# Import constants and utilities from the original script
from generate_wing import (
    Z_OFFSET_OF_BLADES_FOR_BOOLEAN,
    TIP_FILLET_SIZE_REDUCTION,
    TIP_FILLET_EXTENSION_FACTOR,
    TIP_FILLET_REDUCTION_EXPONENT,
    ROOT_FILLET_BLEND_EXPONENT,
    MAX_NACA_THICKNESS,
    MAX_NACA_CAMBER,
    ROOT_BLEND_MORE_MULTIPLIER,
    WingGenerator as MeshWingGenerator,
    load_params_from_csv
)


class BRepWingGenerator:
    """
    Generates 3D wing geometry from parametric design specifications using B-Rep CAD operations.
    """
    
    # Hub cylinder constants (same as mesh version)
    HUB_RADIUS = 0.00435 / 2.  # Hub cylinder radius in meters
    HUB_HEIGHT = 0.0052   # Hub cylinder height in meters
    HOLE_DIAMETER = 0.00081  # Center hole diameter in meters
    HOLE_RADIUS = HOLE_DIAMETER / 2  # Center hole radius in meters
    
    def __init__(self):
        """Initialize the B-Rep wing generator."""
        self.wing_start_location = Vector(0.0, 0.0, Z_OFFSET_OF_BLADES_FOR_BOOLEAN)
        self.revolve_center = Vector(0.0, 0.0, 0.0)
        self.revolve_axis = Vector(0.0, 1.0, 0.0)
        
    def parse_naca4(self, code: str) -> Tuple[float, float, float]:
        """
        Parse a 4-digit NACA code into its components.
        
        Args:
            code: 4-digit NACA code as string (e.g., "2412")
            
        Returns:
            Tuple of (m, p, t) where:
                m = maximum camber as fraction of chord
                p = position of maximum camber as fraction of chord
                t = maximum thickness as fraction of chord
        """
        s = str(code).strip()
        if len(s) != 4 or not s.isdigit():
            raise ValueError(f"NACA code must be 4 digits, got: {code}")
        
        m = int(s[0]) / 100.0  # Maximum camber
        p = int(s[1]) / 10.0   # Position of maximum camber
        t = int(s[2:]) / 100.0 # Maximum thickness
        
        return m, p, t
    
    def generate_naca4_profile(self, m: float, p: float, t: float, n_points: int = 50) -> np.ndarray:
        """
        Generate a NACA 4-digit airfoil profile.
        
        Args:
            m: Maximum camber as fraction of chord
            p: Position of maximum camber as fraction of chord
            t: Maximum thickness as fraction of chord
            n_points: Number of points per side (total points = 2*n_points)
            
        Returns:
            Array of shape (N, 2) with (x, y) coordinates forming a closed loop
        """
        # Generate x coordinates with cosine spacing for better leading edge resolution
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1.0 - np.cos(beta))
        
        # NACA thickness distribution
        yt = 5.0 * t * (
            0.2969 * np.sqrt(x) - 
            0.1260 * x - 
            0.3516 * x**2 + 
            0.2843 * x**3 - 
            0.1015 * x**4
        )
        
        # Camber line and gradient
        if m == 0 or p == 0:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.where(
                x < p,
                (m / (p * p)) * (2.0 * p * x - x * x),
                (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x)
            )
            dyc_dx = np.where(
                x < p,
                (2.0 * m / (p * p)) * (p - x),
                (2.0 * m / ((1.0 - p) ** 2)) * (p - x)
            )
        
        # Angle of camber line
        theta_c = np.arctan(dyc_dx)
        
        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta_c)
        yu = yc + yt * np.cos(theta_c)
        xl = x + yt * np.sin(theta_c)
        yl = yc - yt * np.cos(theta_c)
        
        # Create closed loop: upper surface (reversed) + lower surface
        upper = np.column_stack([xu[::-1], yu[::-1]])
        lower = np.column_stack([xl[1:], yl[1:]])  # Skip first point to avoid duplicate
        
        profile = np.vstack([upper, lower])
        return profile
    
    def create_airfoil_wire(self, profile: np.ndarray, z_pos: float, chord: float, 
                           twist_deg: float, start_location: Vector) -> Wire:
        """
        Create a B-Rep wire from a 2D airfoil profile at a specific spanwise location.
        
        Args:
            profile: Array of shape (N, 2) with normalized (x, y) coordinates
            z_pos: Spanwise position (along Z axis)
            chord: Chord length at this section
            twist_deg: Twist angle in degrees
            start_location: Starting location (x, y, z) of the wing root
            
        Returns:
            Wire object representing the airfoil section
        """
        # Scale and center the profile
        # Leading edge (x=0) becomes x=0.25*chord (positive X)
        # Trailing edge (x=1) becomes x=-0.75*chord (negative X)
        x = chord * (0.25 - profile[:, 0])
        y = chord * profile[:, 1]
        z = np.full_like(x, z_pos)
        
        # Create 3D coordinates
        coords = np.column_stack([x, y, z])
        
        # Apply twist rotation around Z axis at the section location
        twist_rad = math.radians(twist_deg)
        cos_t = math.cos(twist_rad)
        sin_t = math.sin(twist_rad)
        
        # Rotation matrix for Z axis
        rot_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
        
        # Rotate around the section center
        section_center = np.array([0, 0, z_pos])
        coords = (coords - section_center) @ rot_matrix.T + section_center
        
        # Translate to start location
        coords += np.array([start_location.X, start_location.Y, start_location.Z])
        
        # Convert numpy array to list of Vector points
        points = [Vector(c[0], c[1], c[2]) for c in coords]
        
        # Create a closed wire - ensure we explicitly close the loop
        # Add first point at the end to ensure closure
        if len(points) > 0:
            # Check if already closed (first and last points are same)
            first = points[0]
            last = points[-1]
            dist = (first - last).length
            if dist > 1e-9:  # Not closed, add closing point
                points.append(first)
        
        with BuildLine() as line_builder:
            Polyline(*points)
        
        wire = line_builder.wire()
        
        # Validate wire
        if not wire.is_valid:
            raise ValueError(f"Invalid wire created at z={z_pos}")
        if not wire.is_closed:
            # Try to make a wire from edges
            edges = line_builder.edges()
            print(f"Warning: Wire not closed at z={z_pos}, {len(edges)} edges, attempting to close...")
            try:
                wire = Wire.make_wire(edges)
                if not wire.is_closed:
                    raise ValueError(f"Still not closed after make_wire at z={z_pos}")
            except Exception as e:
                raise ValueError(f"Failed to create closed wire at z={z_pos}: {e}")
        
        return wire
    
    def generate_wing_solid(self, params: Dict, n_blend_sections: int = 6,
                          n_profile_points: int = 50,
                          envelope_offset: float = 0.03,
                          n_tip_fillet_sections: int = 5,
                          root_fillet_scale: float = 7.0) -> Solid:
        """
        Generate a single wing as a B-Rep solid using lofting.
        
        Args:
            params: Dictionary with wing parameters
            n_blend_sections: Number of interpolated sections between defined stations
            n_profile_points: Number of points per airfoil side
            envelope_offset: Envelope offset as fraction of chord (not implemented for B-Rep)
            n_tip_fillet_sections: Number of fillet sections at tip
            root_fillet_scale: Scale factor for root section thickness
            
        Returns:
            Solid object representing the wing
        """
        # Extract parameters
        overall_length = params['overall_length']
        n_sections = params['n_sections']
        
        # Get NACA codes, chord lengths, and twist angles
        naca_codes = [params[f'naca_{i}'] for i in range(n_sections)]
        chord_lengths = [params[f'chord_{i}'] for i in range(n_sections)]
        twist_angles = [params[f'twist_{i}'] for i in range(n_sections)]
        
        # Modify root NACA code for fillet effect
        naca_codes_modified = naca_codes.copy()
        if root_fillet_scale > 1.0:
            m_root, p_root, t_root = self.parse_naca4(naca_codes[0])
            t_root_scaled = min(t_root * root_fillet_scale, MAX_NACA_THICKNESS)
            m_root_scaled = min(m_root * root_fillet_scale, MAX_NACA_CAMBER)
            naca_codes_modified[0] = f"{int(m_root_scaled * 100)}{int(p_root * 10)}{int(t_root_scaled * 100):02d}"
        
        # Create spanwise positions
        section_positions = np.linspace(0, overall_length, n_sections)
        
        # Create interpolators for smooth variation
        chord_interpolator = interpolate.interp1d(
            section_positions, chord_lengths, kind='cubic',
            fill_value='extrapolate', assume_sorted=True
        )
        twist_interpolator = interpolate.interp1d(
            section_positions, twist_angles, kind='cubic',
            fill_value='extrapolate', assume_sorted=True
        )
        
        # Generate all section wires
        wires = []
        
        # Add control sections with blended intermediate sections
        for i in range(n_sections - 1):
            # Add the control section
            z_pos = section_positions[i]
            m, p, t = self.parse_naca4(naca_codes_modified[i])
            profile = self.generate_naca4_profile(m, p, t, n_profile_points)
            chord = chord_lengths[i]
            twist = twist_angles[i]
            
            wire = self.create_airfoil_wire(profile, z_pos, chord, twist, self.wing_start_location)
            wires.append(wire)
            
            # Add blended sections between control points
            n_blend = n_blend_sections
            if i == 0:  # More blending near root for fillet
                n_blend = n_blend_sections * ROOT_BLEND_MORE_MULTIPLIER
            
            for k in range(1, n_blend + 1):
                alpha = k / (n_blend + 1)
                z_blend = section_positions[i] + alpha * (section_positions[i + 1] - section_positions[i])
                
                # Blend NACA parameters
                m_a, p_a, t_a = self.parse_naca4(naca_codes_modified[i])
                m_b, p_b, t_b = self.parse_naca4(naca_codes_modified[i + 1])
                
                if i == 0:  # Use power curve blending for root fillet
                    alpha_curved = alpha ** ROOT_FILLET_BLEND_EXPONENT
                    m_blend = (1 - alpha_curved) * m_a + alpha_curved * m_b
                    p_blend = (1 - alpha_curved) * p_a + alpha_curved * p_b
                    t_blend = (1 - alpha_curved) * t_a + alpha_curved * t_b
                else:  # Linear blending
                    m_blend = (1 - alpha) * m_a + alpha * m_b
                    p_blend = (1 - alpha) * p_a + alpha * p_b
                    t_blend = (1 - alpha) * t_a + alpha * t_b
                
                profile_blend = self.generate_naca4_profile(m_blend, p_blend, t_blend, n_profile_points)
                chord_blend = float(chord_interpolator(z_blend))
                twist_blend = float(twist_interpolator(z_blend))
                
                wire_blend = self.create_airfoil_wire(profile_blend, z_blend, chord_blend, 
                                                     twist_blend, self.wing_start_location)
                wires.append(wire_blend)
        
        # Add final control section
        z_pos = section_positions[-1]
        m, p, t = self.parse_naca4(naca_codes_modified[-1])
        profile = self.generate_naca4_profile(m, p, t, n_profile_points)
        chord = chord_lengths[-1]
        twist = twist_angles[-1]
        wire = self.create_airfoil_wire(profile, z_pos, chord, twist, self.wing_start_location)
        wires.append(wire)
        
        # Add tip fillet sections
        if n_tip_fillet_sections > 0:
            last_m, last_p, last_t = self.parse_naca4(naca_codes[-1])
            last_chord = chord_lengths[-1]
            last_twist = twist_angles[-1]
            last_z_pos = section_positions[-1]
            
            final_size_factor = 1.0 - TIP_FILLET_SIZE_REDUCTION
            fillet_extension = last_chord * TIP_FILLET_EXTENSION_FACTOR
            
            for k in range(1, n_tip_fillet_sections + 1):
                alpha = k / (n_tip_fillet_sections + 1)
                z_fillet = last_z_pos + alpha * fillet_extension
                
                alpha_curved = alpha ** TIP_FILLET_REDUCTION_EXPONENT
                reduction_factor = 1.0 - alpha_curved * (1.0 - final_size_factor)
                
                fillet_t = last_t * reduction_factor
                fillet_m = last_m * reduction_factor
                fillet_p = last_p
                
                profile_fillet = self.generate_naca4_profile(fillet_m, fillet_p, fillet_t, n_profile_points)
                chord_fillet = last_chord * reduction_factor
                twist_fillet = last_twist
                
                wire_fillet = self.create_airfoil_wire(profile_fillet, z_fillet, chord_fillet,
                                                       twist_fillet, self.wing_start_location)
                wires.append(wire_fillet)
        
        # Create lofted solid from wires
        # Use Solid.make_loft directly
        print(f"Creating loft with {len(wires)} wires...")
        lofted_solid = Solid.make_loft(wires, ruled=False)
        
        return lofted_solid
    
    def create_hub_with_hole(self) -> Solid:
        """
        Create a cylindrical hub with a central hole using B-Rep primitives.
        
        Returns:
            Solid object representing the hub with hole
        """
        with BuildPart() as hub_builder:
            # Create hub cylinder along Y axis
            with BuildSketch(Plane.YZ) as sketch:
                Circle(self.HUB_RADIUS)
            extrude(amount=self.HUB_HEIGHT, both=True)
            
            # Create central hole
            with BuildSketch(Plane.YZ) as hole_sketch:
                Circle(self.HOLE_RADIUS)
            extrude(amount=self.HUB_HEIGHT * 1.5, both=True, mode=Mode.SUBTRACT)
        
        hub_solid = hub_builder.part
        
        return hub_solid
    
    def generate_complete_design(self, params: Dict, n_blend_sections: int = 6,
                               n_profile_points: int = 50,
                               envelope_offset: float = 0.03,
                               n_tip_fillet_sections: int = 5,
                               root_fillet_scale: float = 7.0) -> Solid:
        """
        Generate complete rotor design with multiple wings and hub.
        
        Args:
            params: Dictionary with wing parameters
            n_blend_sections: Number of interpolated sections between defined stations
            n_profile_points: Number of points per airfoil side
            envelope_offset: Envelope offset as fraction of chord (not implemented for B-Rep)
            n_tip_fillet_sections: Number of fillet sections at tip
            root_fillet_scale: Scale factor for root section thickness
            
        Returns:
            Solid object representing the complete rotor assembly
        """
        n_wings = params['n_wings']
        
        # Generate base wing
        print("Generating base wing solid...")
        base_wing = self.generate_wing_solid(
            params, n_blend_sections, n_profile_points,
            envelope_offset, n_tip_fillet_sections, root_fillet_scale
        )
        
        # Create hub with hole
        print("Creating hub with hole...")
        hub_solid = self.create_hub_with_hole()
        
        # Create rotated wings
        wing_solids = []
        angle_per_wing = 360.0 / n_wings
        
        for i in range(n_wings):
            print(f"Creating wing {i+1}/{n_wings}...")
            angle = angle_per_wing * i
            
            # Rotate wing around Y axis
            rotated_wing = base_wing.rotate(
                axis=Axis(origin=self.revolve_center, direction=self.revolve_axis),
                angle=angle
            )
            wing_solids.append(rotated_wing)
        
        # Boolean union all parts
        print("Performing Boolean union...")
        combined_solid = hub_solid
        for wing in wing_solids:
            combined_solid = combined_solid + wing
        
        return combined_solid


def main():
    """Main function for B-Rep wing generation."""
    parser = argparse.ArgumentParser(
        description='Generate wing geometry from CSV parameters using B-Rep CAD operations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('csv_file', help='Path to CSV file with wing parameters')
    parser.add_argument('--row', type=int, default=0, help='Row index to use (default: 0)')
    parser.add_argument('--output', default='wing_brep', help='Output file base name (default: wing_brep)')
    parser.add_argument('--blend-sections', type=int, default=6,
                       help='Number of blend sections between defined stations (default: 6)')
    parser.add_argument('--profile-points', type=int, default=50,
                       help='Number of points per airfoil side (default: 50)')
    parser.add_argument('--envelope-offset', type=float, default=0.03,
                       help='Envelope offset as fraction of chord (default: 0.03). '
                            'Note: Not implemented for B-Rep version.')
    parser.add_argument('--tip-fillet-sections', type=int, default=5,
                       help='Number of additional tip fillet sections (default: 5).')
    parser.add_argument('--root-fillet-scale', type=float, default=7.0,
                       help='Scale factor for root section thickness (default: 7).')
    
    args = parser.parse_args()
    
    # Load parameters
    print(f"Loading parameters from {args.csv_file}, row {args.row}...")
    params = load_params_from_csv(args.csv_file, args.row)
    
    print(f"Design parameters:")
    print(f"  Overall length: {params['overall_length']:.6f} m")
    print(f"  Number of wings: {params['n_wings']}")
    print(f"  Number of sections: {params['n_sections']}")
    print(f"  Tip fillet sections: {args.tip_fillet_sections}")
    print(f"  Root fillet scale: {args.root_fillet_scale:.2f}")
    
    # Generate wing using B-Rep
    print("\nGenerating B-Rep wing geometry...")
    generator = BRepWingGenerator()
    rotor_solid = generator.generate_complete_design(
        params,
        n_blend_sections=args.blend_sections,
        n_profile_points=args.profile_points,
        envelope_offset=args.envelope_offset,
        n_tip_fillet_sections=args.tip_fillet_sections,
        root_fillet_scale=args.root_fillet_scale
    )
    
    print(f"\nGenerated B-Rep solid successfully!")
    
    # Export B-Rep as STEP file (in millimeters)
    step_file = f"{args.output}.step"
    print(f"Exporting B-Rep to {step_file}...")
    
    # Scale solid from meters to millimeters for export
    rotor_solid_mm = rotor_solid.scale(1000.0)
    
    export_step(rotor_solid_mm, step_file, unit=Unit.MM)
    
    # Convert to mesh and export as STL (also in millimeters)
    stl_file = f"{args.output}.stl"
    print(f"Converting to mesh and exporting to {stl_file}...")
    
    # Export STL (already scaled to mm)
    export_stl(rotor_solid_mm, stl_file, tolerance=0.001, angular_tolerance=0.1)
    
    print("\nDone!")
    print(f"  B-Rep file: {step_file}")
    print(f"  Mesh file: {stl_file}")


if __name__ == '__main__':
    main()
