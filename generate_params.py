#!/usr/bin/env python3
"""
Parameter CSV Generation Script

This script generates a CSV file with wing design parameters from abstract design requirements.
It translates high-level design specifications into the detailed parameter format used by the wing generator.
"""

import argparse
import csv
import math
from typing import List


# NACA airfoil constants
MAX_POSITION_DIGIT = 9  # Maximum value for NACA position digit

# Chord length constants
# ROOT_CHORD_LENGTH is fixed at 1.7 * HUB_RADIUS to ensure proper fit within hub geometry
# HUB_RADIUS = 0.00435 / 2.0 = 0.002175, so 1.7 * 0.002175 = 0.0036975
ROOT_CHORD_LENGTH = 0.0036975  # Fixed chord length at root for rotor hub union (1.7 * HUB_RADIUS)


def translate_to_naca_code(max_thickness: float, max_camber: float, max_camber_location: float) -> str:
    """
    Translate chord max thickness, camber, and location to a NACA 4-digit code.
    
    Args:
        max_thickness: Maximum thickness as a percentage (e.g., 9.0 for 9%)
        max_camber: Maximum camber as a percentage (e.g., 6.0 for 6%)
        max_camber_location: Location parameter [0,1] for position of maximum camber
                           0 = forward (leading edge bias)
                           0.5 = balanced (typical)
                           1.0 = aft (trailing edge bias)
    
    Returns:
        4-digit NACA code as string (e.g., "6409")
    
    Note:
        NACA 4-digit format is MPTT where:
        - M: Maximum camber as percentage of chord (first digit)
        - P: Position of maximum camber in tenths of chord (second digit)
        - TT: Maximum thickness as percentage of chord (last two digits)
        
        The thickness distribution is determined by the NACA formula and has a
        fixed shape, while P controls camber position which affects overall geometry.
    """
    # Maximum camber for wing designs - clamp to valid single digit range
    m = int(round(max_camber))
    m = max(0, min(9, m))  # Ensure 0-9 range for valid NACA codes
    
    # Map location [0,1] to position digit [0,9]
    # This controls the position of maximum camber along the chord
    # Typical values: 3-5 (30%-50% chord)
    p = int(round(max_camber_location * MAX_POSITION_DIGIT))
    p = max(0, min(9, p))  # Ensure 0-9 range
    
    # Thickness: convert to two-digit format and clamp to valid range
    tt = int(round(max_thickness))
    tt = max(1, min(99, tt))  # Ensure 01-99 range
    
    return f"{m}{p}{tt:02d}"


def generate_chord_lengths(average_chord: float, chord_variance: float, n_sections: int = 6) -> List[float]:
    """
    Generate chord lengths for each section based on average and variance.
    
    Args:
        average_chord: Target average chord length for non-root sections (meters).
                      The root section is always ROOT_CHORD_LENGTH regardless of this value.
        chord_variance: Variance [0,1] indicating expansion/shrink pattern
                       0 = constant chord (all non-root sections equal average_chord)
                       1 = maximum variation (expand then shrink with smooth transitions)
    
    Returns:
        List of chord lengths for each section, with root always at ROOT_CHORD_LENGTH
    """
    if n_sections < 2:
        return [ROOT_CHORD_LENGTH]
    
    chord_lengths = []
    
    # First section is always the fixed root chord
    chord_lengths.append(ROOT_CHORD_LENGTH)
    
    # Generate smooth curve for remaining sections
    # Use a combination of cosine functions for smooth transitions
    for i in range(1, n_sections):
        # Normalized position [0, 1] for sections after root
        t = i / (n_sections - 1)
        
        # Create a smooth asymmetric bell curve using cosine-based interpolation
        # This creates a smooth curve that peaks around t=0.33, then gradually decays
        
        # Use a modified raised cosine for smooth transitions
        # Peak around 30-40% of span, then gradual decay with steeper drop at the end
        if t < 0.33:
            # Rising phase: smooth acceleration from root to peak
            # Use raised cosine for smooth S-curve
            phase = t / 0.33
            factor = 0.75 + 1.25 * (1.0 - math.cos(phase * math.pi)) / 2.0
        elif t < 0.7:
            # Gradual decay: smooth transition from peak to mid-span
            phase = (t - 0.33) / 0.37
            factor = 2.0 - 0.75 * (1.0 - math.cos(phase * math.pi)) / 2.0
        else:
            # Steeper drop at tip: smooth but faster decay
            phase = (t - 0.7) / 0.3
            factor = 1.25 - 1.0 * (1.0 - math.cos(phase * math.pi)) / 2.0
        
        # Scale by variance: higher variance means more pronounced variation
        # At variance=0, factor should be 1.0 for all sections
        deviation = (factor - 1.0) * chord_variance
        chord = average_chord * (1.0 + deviation)
        
        # Ensure positive chord length
        chord = max(chord, average_chord * 0.1)
        chord_lengths.append(chord)
    
    return chord_lengths


def generate_twist_angles(max_twist: float, n_sections: int = 6) -> List[float]:
    """
    Generate twist angles for each section.
    
    Args:
        max_twist: Maximum twist angle at the root (degrees)
        n_sections: Number of sections
    
    Returns:
        List of twist angles, with max at root, 0 at tip, interpolated between
    """
    if n_sections < 2:
        return [max_twist]
    
    twist_angles = []
    for i in range(n_sections):
        # Linear interpolation from max_twist to 0
        t = i / (n_sections - 1)
        twist = max_twist * (1.0 - t)
        twist_angles.append(twist)
    
    # Ensure last element is exactly 0
    twist_angles[-1] = 0.0
    
    return twist_angles


def generate_params_csv(
    output_file: str,
    overall_length: float = 0.02,
    chord_max_thickness: float = 9.0,
    max_camber: float = 6.0,
    max_camber_location: float = 0.4,
    average_chord_length: float = 0.002,
    chord_length_variance: float = 0.5,
    max_twist_angle: float = 40.0,
    n_wings: int = 3,
    rpm: float = 3000.0,
    rho: float = 1.225,
    n_sections: int = 6,
    case_index: int = 0
):
    """
    Generate a CSV file with wing design parameters from abstract requirements.
    
    Args:
        output_file: Path to the output CSV file
        overall_length: Overall length of the wing (m)
        chord_max_thickness: Maximum thickness as percentage (e.g., 9.0 for 9%)
        max_camber: Maximum camber as percentage (e.g., 6.0 for 6%)
        max_camber_location: Location of max camber [0,1]
        average_chord_length: Average chord length (m)
        chord_length_variance: Variance in chord length [0,1]
        max_twist_angle: Maximum twist angle at root (degrees)
        n_wings: Number of wings
        rpm: Rotations per minute
        rho: Air density (kg/m^3)
        n_sections: Number of sections along the wing
        case_index: Case index for tracking
    """
    # Translate abstract requirements to concrete parameters
    naca_code = translate_to_naca_code(chord_max_thickness, max_camber, max_camber_location)
    chord_lengths = generate_chord_lengths(average_chord_length, chord_length_variance, n_sections)
    twist_angles = generate_twist_angles(max_twist_angle, n_sections)
    
    # All sections use the same NACA code (root fillet will be handled by generate_wing.py)
    naca_codes = [naca_code] * n_sections
    
    # Prepare the CSV row
    header = ['case_index', 'overall_length', 'n_wings']
    row = [case_index, overall_length, n_wings]
    
    # Add chord lengths
    for i in range(n_sections):
        header.append(f'chord_{i}')
        row.append(chord_lengths[i])
    
    # Add NACA codes
    for i in range(n_sections):
        header.append(f'naca_{i}')
        row.append(naca_codes[i])
    
    # Add twist angles
    for i in range(n_sections):
        header.append(f'twist_{i}')
        row.append(twist_angles[i])
    
    # Add remaining parameters
    header.extend(['rpm', 'rho', 'n_sections'])
    row.extend([rpm, rho, n_sections])
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    
    print(f"Generated parameter file: {output_file}")
    print(f"  Overall length: {overall_length}")
    print(f"  Number of wings: {n_wings}")
    print(f"  NACA code (all sections): {naca_code} (camber={max_camber}%, camber_location={max_camber_location}, thickness={chord_max_thickness}%)")
    print(f"  Chord lengths: {[f'{c:.6f}' for c in chord_lengths]}")
    print(f"  Twist angles: {[f'{t:.1f}' for t in twist_angles]}")
    print(f"  RPM: {rpm}, Density: {rho} kg/m³")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate wing design parameters CSV from abstract requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with all defaults (similar to sample_params.csv)
  python generate_params.py input.csv
  
  # Custom overall length and number of wings
  python generate_params.py input.csv --overall-length 0.03 --n-wings 4
  
  # Custom chord thickness, camber, and camber location
  python generate_params.py input.csv --chord-max-thickness 12 --max-camber 5 --max-camber-location 0.3
  
  # Custom chord distribution
  python generate_params.py input.csv --average-chord-length 0.0025 --chord-length-variance 0.8
  
  # Custom twist angle
  python generate_params.py input.csv --max-twist-angle 45
        """
    )
    
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--overall-length', type=float, default=0.02,
                       help='Overall length of the wing in meters (default: 0.02)')
    parser.add_argument('--chord-max-thickness', type=float, default=9.0,
                       help='Maximum chord thickness as percentage (default: 9.0)')
    parser.add_argument('--max-camber', type=float, default=6.0,
                       help='Maximum camber as percentage (default: 6.0)')
    parser.add_argument('--max-camber-location', type=float, default=0.4,
                       help='Location of max camber [0,1], 0=leading edge (default: 0.4)')
    parser.add_argument('--average-chord-length', type=float, default=0.002,
                       help='Average chord length in meters (default: 0.002)')
    parser.add_argument('--chord-length-variance', type=float, default=0.5,
                       help='Chord length variance [0,1], 0=constant, 1=max variation (default: 0.5)')
    parser.add_argument('--max-twist-angle', type=float, default=40.0,
                       help='Maximum twist angle at root in degrees (default: 40.0)')
    parser.add_argument('--n-wings', type=int, default=3,
                       help='Number of wings (default: 3)')
    parser.add_argument('--rpm', type=float, default=3000.0,
                       help='Rotations per minute (default: 3000.0)')
    parser.add_argument('--rho', type=float, default=1.225,
                       help='Air density in kg/m³ (default: 1.225)')
    parser.add_argument('--n-sections', type=int, default=6,
                       help='Number of sections along the wing (default: 6)')
    parser.add_argument('--case-index', type=int, default=0,
                       help='Case index for tracking (default: 0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.max_camber < 0 or args.max_camber > 9:
        parser.error("max-camber must be between 0 and 9 for valid NACA codes")
    if args.max_camber_location < 0 or args.max_camber_location > 1:
        parser.error("max-camber-location must be between 0 and 1")
    if args.chord_length_variance < 0 or args.chord_length_variance > 1:
        parser.error("chord-length-variance must be between 0 and 1")
    if args.n_sections < 2:
        parser.error("n-sections must be at least 2")
    
    # Generate the CSV
    generate_params_csv(
        output_file=args.output,
        overall_length=args.overall_length,
        chord_max_thickness=args.chord_max_thickness,
        max_camber=args.max_camber,
        max_camber_location=args.max_camber_location,
        average_chord_length=args.average_chord_length,
        chord_length_variance=args.chord_length_variance,
        max_twist_angle=args.max_twist_angle,
        n_wings=args.n_wings,
        rpm=args.rpm,
        rho=args.rho,
        n_sections=args.n_sections,
        case_index=args.case_index
    )


if __name__ == '__main__':
    main()
