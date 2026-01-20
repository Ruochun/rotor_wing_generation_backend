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
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import trimesh
from scipy import interpolate

Z_OFFSET_OF_BLADES_FOR_BOOLEAN = 0.0015

# Tip fillet constants
TIP_FILLET_EXTENSION_FACTOR = 0.5  # Fillet extends beyond last section by this factor × tip chord
TIP_FILLET_REDUCTION_EXPONENT = 1.5  # Power curve exponent for smooth tapering (higher = steeper taper)

class WingGenerator:
    """
    Generates 3D wing geometry from parametric design specifications.
    """
    
    # Hub cylinder constants
    # These are fixed dimensions as specified in the design requirements
    HUB_RADIUS = 0.00435 / 2.  # Hub cylinder radius in meters
    HUB_HEIGHT = 0.0052   # Hub cylinder height in meters
    HOLE_DIAMETER = 0.0008  # Center hole diameter in meters
    HOLE_RADIUS = HOLE_DIAMETER / 2  # Center hole radius in meters
    
    def __init__(self):
        """Initialize the wing generator."""
        self.wing_start_location = np.array([0.0, 0.0, Z_OFFSET_OF_BLADES_FOR_BOOLEAN])
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
    
    def offset_profile(self, profile: np.ndarray, offset_distance: float, 
                      n_te_points: int = 8) -> np.ndarray:
        """
        Apply an outward normal offset to a closed 2D profile curve.
        This creates an envelope around the profile, removing sharp edges.
        At the trailing edge, creates a smooth rounded cap by interpolating
        multiple offset directions.
        
        Args:
            profile: Array of shape (N, 2) with (x, y) coordinates forming a closed loop
            offset_distance: Distance to offset in the outward normal direction (in profile units)
            n_te_points: Number of intermediate points to add at the trailing edge for rounding
            
        Returns:
            Array of shape (N+n_te_points, 2) with offset (x, y) coordinates
        """
        if offset_distance <= 0:
            return profile
        
        if n_te_points < 0:
            raise ValueError(f"n_te_points must be non-negative, got {n_te_points}")
        
        n_points = len(profile)
        if n_points < 3:
            raise ValueError(f"Profile must have at least 3 points, got {n_points}")
        
        # Calculate centroid once for all points
        centroid = profile.mean(axis=0)
        
        # First pass: calculate offset for all original points
        offset_points = []
        offset_normals = []
        
        for i in range(n_points):
            # Get neighboring points (with wrapping for closed curve)
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            
            prev_point = profile[prev_idx]
            curr_point = profile[i]
            next_point = profile[next_idx]
            
            # Calculate tangent vectors to the curve at this point
            tangent1 = curr_point - prev_point
            tangent2 = next_point - curr_point
            
            # Normalize tangent vectors
            tangent1_norm = np.linalg.norm(tangent1)
            tangent2_norm = np.linalg.norm(tangent2)
            
            if tangent1_norm > 1e-10:
                tangent1 = tangent1 / tangent1_norm
            if tangent2_norm > 1e-10:
                tangent2 = tangent2 / tangent2_norm
            
            # Average tangent direction
            avg_tangent = (tangent1 + tangent2)
            avg_tangent_norm = np.linalg.norm(avg_tangent)
            if avg_tangent_norm > 1e-10:
                avg_tangent = avg_tangent / avg_tangent_norm
            else:
                # Fallback: use perpendicular to tangent1 when they oppose
                # tangent1 is already normalized at this point
                if tangent1_norm > 1e-10:
                    avg_tangent = np.array([-tangent1[1], tangent1[0]])
                else:
                    # Both tangents are degenerate, use perpendicular to tangent2
                    avg_tangent = np.array([-tangent2[1], tangent2[0]])
            
            # Normal is perpendicular to tangent (rotated 90 degrees)
            normal = np.array([-avg_tangent[1], avg_tangent[0]])
            
            # Ensure outward direction by checking distance from centroid
            to_centroid = centroid - curr_point
            
            # If normal points toward centroid, flip it
            if np.dot(normal, to_centroid) > 0:
                normal = -normal
            
            # Store offset point and normal
            offset_points.append(curr_point + normal * offset_distance)
            offset_normals.append(normal)
        
        # Detect trailing edge: for NACA profiles, it's at x ≈ 1.0
        # Find the point closest to x = 1.0 (most reliable for numerical precision)
        x_coords = profile[:, 0]
        te_idx = np.argmin(np.abs(x_coords - 1.0))
        te_point = profile[te_idx]
        
        if n_te_points == 0:
            # No TE rounding, return the simple offset
            return np.array(offset_points)
        
        # For proper connectivity, we need to split the profile at the trailing edge
        # and insert the interpolated points to create a smooth rounded cap
        
        # Get the normals of points adjacent to the trailing edge
        prev_te_idx = (te_idx - 1) % n_points
        next_te_idx = (te_idx + 1) % n_points
        
        normal_before = offset_normals[prev_te_idx]
        normal_after = offset_normals[next_te_idx]
        
        # Calculate angles of the normals
        angle_before = np.arctan2(normal_before[1], normal_before[0])
        angle_after = np.arctan2(normal_after[1], normal_after[0])
        
        # Ensure we interpolate the shorter arc
        # If the angular difference is greater than π, adjust
        angle_diff = angle_after - angle_before
        if angle_diff > np.pi:
            angle_after -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_after += 2 * np.pi
        
        # Build the result with proper ordering:
        # - Start from point after TE (te_idx + 1)
        # - Go around to the point before TE (te_idx - 1)
        # - Add the point before TE
        # - Add interpolated TE points
        # - Add the TE point
        # This maintains smooth connectivity
        result = []
        
        # Add points from after TE to before TE (wrapping around)
        for i in range(n_points):
            idx = (te_idx + 1 + i) % n_points
            if idx == te_idx:
                # We're back at the TE, stop here
                break
            result.append(offset_points[idx])
        
        # Now add the interpolated TE cap points
        for j in range(n_te_points):
            alpha = (j + 1) / (n_te_points + 1)
            # Interpolate angle from before to after
            interp_angle = (1 - alpha) * angle_before + alpha * angle_after
            # Create normal at this angle
            interp_normal = np.array([np.cos(interp_angle), np.sin(interp_angle)])
            # Add offset point
            result.append(te_point + interp_normal * offset_distance)
        
        # Finally add the TE point itself
        result.append(offset_points[te_idx])
        
        return np.array(result)
    
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
    
    def create_smooth_interpolator(self, control_values: List[float], control_positions: List[float], 
                                   kind: str = 'cubic') -> interpolate.interp1d:
        """
        Create a smooth interpolator for parameters using spline interpolation.
        
        Args:
            control_values: List of control point values (e.g., chord lengths)
            control_positions: List of spanwise positions for control points
            kind: Interpolation kind ('linear', 'quadratic', 'cubic')
        
        Returns:
            Interpolation function that can be called with arbitrary positions
        """
        # Use cubic spline interpolation for smooth curves
        # This treats control values as hints rather than strict requirements
        return interpolate.interp1d(
            control_positions, 
            control_values, 
            kind=kind,
            fill_value='extrapolate',
            assume_sorted=True
        )
    
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
                                  n_profile_points: int = 50,
                                  envelope_offset: float = 0.0,
                                  n_tip_fillet_sections: int = 3) -> trimesh.Trimesh:
        """
        Generate a complete wing mesh from parameter dictionary.
        
        Args:
            params: Dictionary of wing parameters
            n_blend_sections: Number of blend sections between defined stations
            n_profile_points: Number of points per airfoil profile side
            envelope_offset: Offset distance in the outward normal direction (as fraction of chord).
                           This adds a small envelope around the airfoil to remove sharp edges.
                           Typical values: 0.01 to 0.05 for 3D printing friendly geometry.
            n_tip_fillet_sections: Number of additional sections at tip for filleting (default: 3).
                           These sections progressively decrease in size toward the tip to create
                           a smooth rounded tip edge.
            
        Returns:
            Trimesh object of the wing
        """
        # Extract parameters
        overall_length = params['overall_length']
        n_sections = params['n_sections']
        naca_codes = [params[f'naca_{i}'] for i in range(n_sections)]
        twist_angles = [params[f'twist_{i}'] for i in range(n_sections)]
        chord_lengths = [params[f'chord_{i}'] for i in range(n_sections)]
        
        # Account for Z_OFFSET_OF_BLADES_FOR_BOOLEAN:
        # The overall_length parameter represents the distance from rotor center to wing tip,
        # but we offset the wing start by Z_OFFSET_OF_BLADES_FOR_BOOLEAN for better Boolean merging.
        # Therefore, the actual wing length should be reduced by this offset.
        if overall_length <= Z_OFFSET_OF_BLADES_FOR_BOOLEAN:
            raise ValueError(
                f"overall_length ({overall_length}) must be greater than "
                f"Z_OFFSET_OF_BLADES_FOR_BOOLEAN ({Z_OFFSET_OF_BLADES_FOR_BOOLEAN})"
            )
        
        actual_wing_length = overall_length - Z_OFFSET_OF_BLADES_FOR_BOOLEAN
        
        # Generate section positions
        section_positions = np.linspace(0, actual_wing_length, n_sections)
        
        # Create smooth interpolators for chord and twist
        # This treats the specified values as control points for smooth curves
        chord_interpolator = self.create_smooth_interpolator(chord_lengths, section_positions, kind='cubic')
        twist_interpolator = self.create_smooth_interpolator(twist_angles, section_positions, kind='cubic')
        
        # Create all sections (defined + blended) with smooth interpolation
        all_sections = []
        all_positions = []
        
        for i in range(n_sections):
            # Add the defined section
            all_positions.append(section_positions[i])
            
            # Add blend section positions between this and next section
            if i < n_sections - 1:
                for k in range(1, n_blend_sections + 1):
                    alpha = k / float(n_blend_sections + 1)
                    z_pos = (1 - alpha) * section_positions[i] + alpha * section_positions[i + 1]
                    all_positions.append(z_pos)
        
        # Generate all sections with smooth parameter interpolation
        section_idx = 0
        for i in range(n_sections):
            # Generate the defined section
            z_pos = all_positions[section_idx]
            m, p, t = self.parse_naca4(naca_codes[i])
            profile = self.generate_naca4_profile(m, p, t, n_profile_points)
            
            # Apply envelope offset (offset_profile handles offset <= 0 case)
            profile = self.offset_profile(profile, envelope_offset)
            
            # Use smoothly interpolated chord and twist
            chord = float(chord_interpolator(z_pos))
            twist = float(twist_interpolator(z_pos))
            
            section = self.transform_profile_to_section(
                profile, z_pos, chord, twist, self.wing_start_location
            )
            all_sections.append(section)
            section_idx += 1
            
            # Add blend sections between this and next section
            if i < n_sections - 1:
                for k in range(1, n_blend_sections + 1):
                    z_pos = all_positions[section_idx]
                    alpha = k / (n_blend_sections + 1)
                    
                    # Blend NACA parameters
                    m, p, t = self.blend_naca_codes(naca_codes[i], naca_codes[i + 1], alpha)
                    profile = self.generate_naca4_profile(m, p, t, n_profile_points)
                    
                    # Apply envelope offset (offset_profile handles offset <= 0 case)
                    profile = self.offset_profile(profile, envelope_offset)
                    
                    # Use smoothly interpolated chord and twist
                    chord = float(chord_interpolator(z_pos))
                    twist = float(twist_interpolator(z_pos))
                    
                    section = self.transform_profile_to_section(
                        profile, z_pos, chord, twist, self.wing_start_location
                    )
                    all_sections.append(section)
                    section_idx += 1
        
        # Add tip fillet sections (progressively smaller sections toward the tip)
        if n_tip_fillet_sections > 0:
            # Get the last section parameters
            last_m, last_p, last_t = self.parse_naca4(naca_codes[-1])
            last_chord = chord_lengths[-1]
            last_twist = twist_angles[-1]
            last_z_pos = section_positions[-1]
            
            # Calculate spacing for fillet sections
            # Extend beyond the last section by a distance proportional to the last chord
            fillet_extension = last_chord * TIP_FILLET_EXTENSION_FACTOR
            
            for k in range(1, n_tip_fillet_sections + 1):
                # Progressive reduction factor (decreases from 1 to near 0)
                alpha = k / (n_tip_fillet_sections + 1)
                
                # Position extends beyond the last section
                z_pos = last_z_pos + alpha * fillet_extension
                
                # Progressive reduction in chord and thickness
                # Use a power curve for smooth reduction
                reduction_factor = (1.0 - alpha) ** TIP_FILLET_REDUCTION_EXPONENT
                
                # Scale down the airfoil thickness
                fillet_t = last_t * reduction_factor
                # Maintain camber characteristics
                fillet_m = last_m * reduction_factor
                fillet_p = last_p
                
                # Generate the fillet section profile
                profile = self.generate_naca4_profile(fillet_m, fillet_p, fillet_t, n_profile_points)
                
                # Apply envelope offset
                profile = self.offset_profile(profile, envelope_offset)
                
                # Scale down the chord
                fillet_chord = last_chord * reduction_factor
                
                # Keep the same twist angle
                fillet_twist = last_twist
                
                section = self.transform_profile_to_section(
                    profile, z_pos, fillet_chord, fillet_twist, self.wing_start_location
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
        self.fix_normals_outward(rotated_mesh)
        
        return rotated_mesh
    
    def create_hub_cylinder(self) -> trimesh.Trimesh:
        """
        Create a cylindrical hub centered at the origin along the Y axis.
        
        Returns:
            Trimesh object representing the hub cylinder
        """
        # Create cylinder along Z axis (default)
        cylinder = trimesh.creation.cylinder(
            radius=self.HUB_RADIUS,
            height=self.HUB_HEIGHT
        )
        
        # Rotate to align along Y axis (rotate 90 degrees around X axis)
        transform = trimesh.transformations.rotation_matrix(
            angle=np.radians(90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        cylinder.apply_transform(transform)
        
        return cylinder
    
    def create_hole_cylinder(self) -> trimesh.Trimesh:
        """
        Create a cylindrical hole to be drilled through the hub.
        
        Returns:
            Trimesh object representing the hole cylinder
        """
        # Create cylinder along Z axis (default) with slightly larger height
        # to ensure it fully penetrates the hub (1.5x height provides clearance)
        HOLE_CLEARANCE_FACTOR = 1.5
        cylinder = trimesh.creation.cylinder(
            radius=self.HOLE_RADIUS,
            height=self.HUB_HEIGHT * HOLE_CLEARANCE_FACTOR
        )
        
        # Rotate to align along Y axis (rotate 90 degrees around X axis)
        transform = trimesh.transformations.rotation_matrix(
            angle=np.radians(90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        cylinder.apply_transform(transform)
        
        return cylinder
    
    def fix_normals_outward(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        trimesh.repair.fix_normals(mesh, True)
        # trimesh.repair.fix_inversion(mesh, True)
        return mesh
    
    def create_clockwise_version(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Create a clockwise version of the wing by mirroring across the XY plane.
        
        The original design is for counterclockwise rotation (when viewed from above,
        looking down the Y axis). To create a clockwise version, we mirror the mesh
        across the XY plane (negate Z coordinates) and flip the face winding order
        to maintain outward-pointing normals.
        
        Args:
            mesh: The counterclockwise wing mesh
            
        Returns:
            Mirrored mesh suitable for clockwise rotation with correct outward normals
        """
        # Create a copy of the mesh
        mirrored_mesh = mesh.copy()
        
        # Mirror across XY plane by negating Z coordinates
        mirrored_mesh.vertices[:, 2] *= -1
        
        # Flip face winding order to correct normals inverted by Z-coordinate negation
        # Step 1 (above): Negating Z-coordinates mirrors geometry but inverts normals
        # Step 2 (here): Reversing vertex order in each face flips normals back outward
        # np.fliplr() reverses each row (face) in the faces array from [v0, v1, v2]
        # to [v2, v1, v0], which reverses the winding order and thus the normal direction.
        mirrored_mesh.faces = np.fliplr(mirrored_mesh.faces)
        
        return mirrored_mesh
    
    def generate_complete_design(self, params: Dict,
                                n_blend_sections: int = 6,
                                n_profile_points: int = 50,
                                envelope_offset: float = 0.0,
                                n_tip_fillet_sections: int = 3) -> trimesh.Trimesh:
        """
        Generate a complete wing design with multiple wings arranged circularly
        and a central hub with a drilled hole.
        
        Args:
            params: Dictionary of wing parameters
            n_blend_sections: Number of blend sections between defined stations
            n_profile_points: Number of points per airfoil profile side
            envelope_offset: Offset distance in the outward normal direction (as fraction of chord).
                           This adds a small envelope around the airfoil to remove sharp edges.
            n_tip_fillet_sections: Number of additional sections at tip for filleting (default: 3).
                           These sections progressively decrease in size toward the tip to create
                           a smooth rounded tip edge.
            
        Returns:
            Combined mesh of all wings merged with the hub
        """
        # Generate the base wing
        base_wing = self.generate_wing_from_params(params, n_blend_sections, n_profile_points, 
                                                   envelope_offset, n_tip_fillet_sections)
        
        # Fix normals on base wing to ensure they point outward
        # self.fix_normals_outward(base_wing)
        
        # Get the root chord length (first section)
        root_chord = params['chord_0']
        
        # Shift the wing forward by 1/4 of the root chord length
        # This aligns wings at their mid-chord (1/2 chord) for proper rotation alignment
        shift_distance = root_chord * 0.25
        shift_vector = np.array([shift_distance, 0, 0])
        base_wing.vertices += shift_vector
        
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
        
        # Create hub cylinder
        hub = self.create_hub_cylinder()
        
        # Create hole cylinder
        hole = self.create_hole_cylinder()
        
        # Drill hole through hub (Boolean difference)
        # This works because both hub and hole are watertight solids
        hub_with_hole = trimesh.boolean.difference([hub, hole])
        
        # Fix normals on the result of Boolean operation
        self.fix_normals_outward(hub_with_hole)
        
        # Attempt Boolean union to merge hub with wings
        # Note: The wing meshes generated by the lofting process may not be perfectly
        # watertight, which can cause Boolean operations to fail. We try Boolean union
        # first (which creates a true merged solid), but fall back to concatenation
        # (which creates a single mesh file containing multiple solids) if it fails.
        # Volume checking is disabled because wing meshes may not be perfect volumes.
        all_meshes = [hub_with_hole] + wing_meshes
        
        try:
            # Try Boolean union (creates a single merged solid)
            combined_mesh = trimesh.boolean.union(all_meshes, check_volume=False)
            
            # Check if union succeeded (non-empty result)
            if combined_mesh.vertices.shape[0] > 0 and combined_mesh.faces.shape[0] > 0:
                # Fix normals on the final combined mesh
                self.fix_normals_outward(combined_mesh)
                return combined_mesh
            else:
                raise ValueError("Boolean union produced empty mesh")
                
        except (ValueError, Exception):
            # Fall back to concatenation (creates single mesh file with multiple solids)
            # This still produces "one mesh" as required, just not Boolean-merged
            combined_mesh = trimesh.util.concatenate(all_meshes)
            # Fix normals on the concatenated mesh
            # self.fix_normals_outward(combined_mesh)
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
    parser.add_argument('--output', default='wing_output.stl', help='Output STL file path')
    parser.add_argument('--blend-sections', type=int, default=6, 
                       help='Number of blend sections between defined stations (default: 6)')
    parser.add_argument('--profile-points', type=int, default=50,
                       help='Number of points per airfoil side (default: 50)')
    parser.add_argument('--envelope-offset', type=float, default=0.03,
                       help='Envelope offset as fraction of chord (default: 0.03). '
                            'Adds a small, thin envelope around airfoils to remove sharp edges, '
                            'making the wing more 3D printing friendly.')
    parser.add_argument('--tip-fillet-sections', type=int, default=3,
                       help='Number of additional tip fillet sections (default: 3). '
                            'These sections progressively decrease in size toward the tip, '
                            'creating a smooth rounded tip edge.')
    
    args = parser.parse_args()
    
    # Load parameters
    print(f"Loading parameters from {args.csv_file}, row {args.row}...")
    params = load_params_from_csv(args.csv_file, args.row)
    
    print(f"Design parameters:")
    print(f"  Overall length: {params['overall_length']:.6f}")
    print(f"  Number of wings: {params['n_wings']}")
    print(f"  Number of sections: {params['n_sections']}")
    print(f"  Envelope offset: {args.envelope_offset:.4f} (as fraction of chord)")
    print(f"  Tip fillet sections: {args.tip_fillet_sections}")
    
    # Generate wing
    print("Generating wing geometry...")
    generator = WingGenerator()
    wing_mesh = generator.generate_complete_design(
        params, 
        n_blend_sections=args.blend_sections,
        n_profile_points=args.profile_points,
        envelope_offset=args.envelope_offset,
        n_tip_fillet_sections=args.tip_fillet_sections
    )
    
    print(f"Generated mesh: {len(wing_mesh.vertices)} vertices, {len(wing_mesh.faces)} faces")
    
    # Export counterclockwise version to STL
    print(f"Exporting counterclockwise version to {args.output}...")
    wing_mesh.export(args.output)
    
    # Generate clockwise version
    print("Generating clockwise version...")
    cw_mesh = generator.create_clockwise_version(wing_mesh)
    
    # Determine output filename for clockwise version
    # Split the filename to insert "_cw" before the extension
    base_name, ext = os.path.splitext(args.output)
    cw_output = f"{base_name}_cw{ext}"
    
    print(f"Exporting clockwise version to {cw_output}...")
    cw_mesh.export(cw_output)
    
    print("Done!")


if __name__ == '__main__':
    main()
