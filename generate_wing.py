#!/usr/bin/env python3
"""
Pure-Python Wing Generation System

This module generates parametric wing designs from CSV parameters and exports them as OBJ files.
It uses only readily available Python libraries (no Rhino dependency).

The system supports the 34-parameter wing design format used in this project:
- Overall length, number of wings, number of sections
- NACA airfoil codes for each section
- Twist angles for each section
- Chord lengths for each section
- Twist and chord range parameters
"""

import csv
import math
import numpy as np
import trimesh
from typing import List, Tuple, Dict, Optional


class WingGenerator:
    """
    Generates 3D wing geometry from parametric design specifications.
    """
    
    def __init__(self):
        """Initialize the wing generator."""
        self.wing_start_location = np.array([0.0, 0.0, 0.001])
        self.revolve_center = np.array([0.0, 0.0, 0.0])
        self.revolve_axis = np.array([0.0, 1.0, 0.0])
        
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
    
    def generate_naca4_profile(self, m: float, p: float, t: float, 
                               n_points: int = 100, 
                               closed_te: bool = True) -> np.ndarray:
        """
        Generate NACA 4-digit airfoil profile coordinates.
        
        Args:
            m: Maximum camber as fraction of chord
            p: Position of maximum camber as fraction of chord
            t: Maximum thickness as fraction of chord
            n_points: Number of points per side (upper/lower)
            closed_te: Whether to close the trailing edge
            
        Returns:
            Array of shape (N, 2) with (x, y) coordinates forming a closed loop
        """
        # Thickness distribution coefficients
        a0, a1, a2, a3 = 0.2969, -0.1260, -0.3516, 0.2843
        a4 = -0.1036 if closed_te else -0.1015
        
        # Cosine spacing for better resolution at leading/trailing edges
        theta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1.0 - np.cos(theta))
        
        # Thickness distribution
        yt = 5.0 * t * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)
        
        # Camber line and gradient
        if m == 0.0 or p == 0.0:
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
    
    def transform_profile_to_section(self, profile: np.ndarray, 
                                     z_pos: float, 
                                     chord: float, 
                                     twist_deg: float,
                                     start_location: np.ndarray) -> np.ndarray:
        """
        Transform a 2D airfoil profile to a 3D section at a specific spanwise location.
        
        Args:
            profile: Array of shape (N, 2) with normalized (x, y) coordinates
            z_pos: Spanwise position (along Z axis)
            chord: Chord length at this section
            twist_deg: Twist angle in degrees
            start_location: Starting location (x, y, z) of the wing root
            
        Returns:
            Array of shape (N, 3) with 3D coordinates
        """
        # Scale and center the profile
        # Leading edge (x=0) becomes x=0.25*chord (positive X)
        # Trailing edge (x=1) becomes x=-0.5*chord (negative X)
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
        coords += start_location
        
        return coords
    
    def blend_naca_codes(self, code_a: str, code_b: str, alpha: float) -> Tuple[float, float, float]:
        """
        Blend two NACA codes for smooth transition.
        
        Args:
            code_a: First NACA code
            code_b: Second NACA code
            alpha: Blending factor (0 = code_a, 1 = code_b)
            
        Returns:
            Tuple of (m, p, t) for the blended airfoil
        """
        m_a, p_a, t_a = self.parse_naca4(code_a)
        m_b, p_b, t_b = self.parse_naca4(code_b)
        
        m = (1 - alpha) * m_a + alpha * m_b
        p = (1 - alpha) * p_a + alpha * p_b
        t = (1 - alpha) * t_a + alpha * t_b
        
        return m, p, t
    
    def create_lofted_surface(self, sections: List[np.ndarray]) -> trimesh.Trimesh:
        """
        Create a lofted surface from multiple airfoil sections.
        
        Args:
            sections: List of arrays, each of shape (N, 3) representing a section
            
        Returns:
            Trimesh object representing the lofted surface
        """
        if len(sections) < 2:
            raise ValueError("Need at least 2 sections to create a lofted surface")
        
        n_points = len(sections[0])
        n_sections = len(sections)
        
        # Verify all sections have the same number of points
        for i, section in enumerate(sections):
            if len(section) != n_points:
                raise ValueError(f"Section {i} has {len(section)} points, expected {n_points}")
        
        # Create vertices
        vertices = []
        for section in sections:
            vertices.extend(section)
        vertices = np.array(vertices)
        
        # Create faces by connecting adjacent sections
        faces = []
        for i in range(n_sections - 1):
            for j in range(n_points):
                j_next = (j + 1) % n_points
                
                # Current section indices
                v0 = i * n_points + j
                v1 = i * n_points + j_next
                
                # Next section indices
                v2 = (i + 1) * n_points + j_next
                v3 = (i + 1) * n_points + j
                
                # Create two triangles for the quad
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
        
        faces = np.array(faces)
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return mesh
    
    def cap_ends(self, mesh: trimesh.Trimesh, sections: List[np.ndarray]) -> trimesh.Trimesh:
        """
        Add planar caps to the root and tip of the wing.
        
        Args:
            mesh: The wing surface mesh
            sections: List of section coordinates used to create the mesh
            
        Returns:
            Mesh with caps added
        """
        meshes = [mesh]
        
        # Cap the root (first section)
        root_section = sections[0]
        root_center = root_section.mean(axis=0)
        
        # Create triangles from center to perimeter
        root_faces = []
        n_points = len(root_section)
        
        # Add center vertex
        root_vertices = [root_center]
        # Add perimeter vertices
        root_vertices.extend(root_section)
        root_vertices = np.array(root_vertices)
        
        for i in range(n_points):
            i_next = (i + 1) % n_points
            # Triangle: center, point i, point i+1
            root_faces.append([0, i + 1, i_next + 1])
        
        root_cap = trimesh.Trimesh(vertices=root_vertices, faces=root_faces)
        meshes.append(root_cap)
        
        # Cap the tip (last section)
        tip_section = sections[-1]
        tip_center = tip_section.mean(axis=0)
        
        tip_faces = []
        n_points = len(tip_section)
        
        # Add center vertex
        tip_vertices = [tip_center]
        # Add perimeter vertices
        tip_vertices.extend(tip_section)
        tip_vertices = np.array(tip_vertices)
        
        for i in range(n_points):
            i_next = (i + 1) % n_points
            # Triangle: center, point i+1, point i (reversed for correct normal)
            tip_faces.append([0, i_next + 1, i + 1])
        
        tip_cap = trimesh.Trimesh(vertices=tip_vertices, faces=tip_faces)
        meshes.append(tip_cap)
        
        # Combine all meshes
        combined = trimesh.util.concatenate(meshes)
        
        return combined
    
    def generate_wing_from_params(self, params: Dict, 
                                  n_blend_sections: int = 6,
                                  n_profile_points: int = 50) -> trimesh.Trimesh:
        """
        Generate a complete wing mesh from parameter dictionary.
        
        Args:
            params: Dictionary of wing parameters
            n_blend_sections: Number of blend sections between defined stations
            n_profile_points: Number of points per airfoil profile side
            
        Returns:
            Trimesh object of the wing
        """
        # Extract parameters
        overall_length = params['overall_length']
        n_sections = params['n_sections']
        naca_codes = [params[f'naca_{i}'] for i in range(n_sections)]
        twist_angles = [params[f'twist_{i}'] for i in range(n_sections)]
        chord_lengths = [params[f'chord_{i}'] for i in range(n_sections)]
        
        # Generate section positions
        section_positions = np.linspace(0, overall_length, n_sections)
        
        # Create all sections (defined + blended)
        all_sections = []
        
        for i in range(n_sections):
            # Generate the defined section
            m, p, t = self.parse_naca4(naca_codes[i])
            profile = self.generate_naca4_profile(m, p, t, n_profile_points)
            section = self.transform_profile_to_section(
                profile, section_positions[i], chord_lengths[i], 
                twist_angles[i], self.wing_start_location
            )
            all_sections.append(section)
            
            # Add blend sections between this and next section
            if i < n_sections - 1:
                for k in range(1, n_blend_sections + 1):
                    alpha = k / float(n_blend_sections + 1)
                    
                    # Blend position, NACA, twist, and chord
                    z_pos = (1 - alpha) * section_positions[i] + alpha * section_positions[i + 1]
                    m, p, t = self.blend_naca_codes(naca_codes[i], naca_codes[i + 1], alpha)
                    twist = (1 - alpha) * twist_angles[i] + alpha * twist_angles[i + 1]
                    chord = (1 - alpha) * chord_lengths[i] + alpha * chord_lengths[i + 1]
                    
                    # Generate blended section
                    profile = self.generate_naca4_profile(m, p, t, n_profile_points)
                    section = self.transform_profile_to_section(
                        profile, z_pos, chord, twist, self.wing_start_location
                    )
                    all_sections.append(section)
        
        # Create lofted surface
        wing_mesh = self.create_lofted_surface(all_sections)
        
        # Add end caps
        wing_mesh = self.cap_ends(wing_mesh, all_sections)
        
        return wing_mesh
    
    def revolve_wing(self, wing_mesh: trimesh.Trimesh, angle_deg: float) -> trimesh.Trimesh:
        """
        Revolve a wing around the Y axis by a specified angle.
        
        Args:
            wing_mesh: The wing mesh to revolve
            angle_deg: Rotation angle in degrees
            
        Returns:
            Rotated wing mesh
        """
        if angle_deg == 0:
            return wing_mesh
        
        # Create rotation matrix around Y axis
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Rotation around Y axis through revolve_center
        rot_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Apply rotation
        vertices = wing_mesh.vertices.copy()
        vertices = (vertices - self.revolve_center) @ rot_matrix.T + self.revolve_center
        
        # Create new mesh with rotated vertices
        rotated_mesh = trimesh.Trimesh(vertices=vertices, faces=wing_mesh.faces)
        
        return rotated_mesh
    
    def generate_complete_design(self, params: Dict,
                                n_blend_sections: int = 6,
                                n_profile_points: int = 50) -> trimesh.Trimesh:
        """
        Generate a complete wing design with multiple wings arranged circularly.
        
        Args:
            params: Dictionary of wing parameters
            n_blend_sections: Number of blend sections between defined stations
            n_profile_points: Number of points per airfoil profile side
            
        Returns:
            Combined mesh of all wings
        """
        # Generate the base wing
        base_wing = self.generate_wing_from_params(params, n_blend_sections, n_profile_points)
        
        # Get number of wings
        n_wings = params['n_wings']
        
        # Calculate angle between wings
        angle_per_wing = 360.0 / n_wings if n_wings > 1 else 0.0
        
        # Create all wings
        wing_meshes = []
        for i in range(n_wings):
            angle = angle_per_wing * i
            rotated_wing = self.revolve_wing(base_wing, angle)
            wing_meshes.append(rotated_wing)
        
        # Combine all wings
        if len(wing_meshes) == 1:
            combined_mesh = wing_meshes[0]
        else:
            combined_mesh = trimesh.util.concatenate(wing_meshes)
        
        return combined_mesh


def load_params_from_csv(csv_file: str, row_index: int = 0) -> Dict:
    """
    Load wing parameters from a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        row_index: Index of the row to load (0-based, excluding header)
        
    Returns:
        Dictionary of parameters
    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if row_index >= len(rows):
            raise ValueError(f"Row index {row_index} out of range (file has {len(rows)} rows)")
        
        row = rows[row_index]
        
        # Convert string values to appropriate types
        params = {}
        excluded_keys = {'case_index', 'valid'}
        
        for key, value in row.items():
            if key in excluded_keys:
                continue
            
            # Try to determine the type
            if key.startswith('naca_'):
                # Keep NACA codes as strings
                params[key] = value
            elif key in ['n_wings', 'n_sections']:
                # Integer parameters
                params[key] = int(float(value))
            else:
                # Float parameters
                params[key] = float(value)
        
        return params


def main():
    """Example usage of the wing generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate wing geometry from CSV parameters')
    parser.add_argument('csv_file', help='Path to CSV file with wing parameters')
    parser.add_argument('--row', type=int, default=0, help='Row index to use (default: 0)')
    parser.add_argument('--output', default='wing_output.obj', help='Output OBJ file path')
    parser.add_argument('--blend-sections', type=int, default=6, 
                       help='Number of blend sections between defined stations (default: 6)')
    parser.add_argument('--profile-points', type=int, default=50,
                       help='Number of points per airfoil side (default: 50)')
    
    args = parser.parse_args()
    
    # Load parameters
    print(f"Loading parameters from {args.csv_file}, row {args.row}...")
    params = load_params_from_csv(args.csv_file, args.row)
    
    print(f"Design parameters:")
    print(f"  Overall length: {params['overall_length']:.6f}")
    print(f"  Number of wings: {params['n_wings']}")
    print(f"  Number of sections: {params['n_sections']}")
    
    # Generate wing
    print("Generating wing geometry...")
    generator = WingGenerator()
    wing_mesh = generator.generate_complete_design(
        params, 
        n_blend_sections=args.blend_sections,
        n_profile_points=args.profile_points
    )
    
    print(f"Generated mesh: {len(wing_mesh.vertices)} vertices, {len(wing_mesh.faces)} faces")
    
    # Export to OBJ
    print(f"Exporting to {args.output}...")
    wing_mesh.export(args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()
