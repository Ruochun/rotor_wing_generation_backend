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


def translate_to_naca_code(max_thickness: float, max_thickness_location: float) -> str:
    """
    Translate chord max thickness and location to a NACA 4-digit code.
    
    Args:
        max_thickness: Maximum thickness as a percentage (e.g., 9.0 for 9%)
        max_thickness_location: Location of max thickness [0,1], where 0 is leading edge, 1 is trailing edge
                              For NACA 4-digit: 0=leading edge, 0.4 is typical location
    
    Returns:
        4-digit NACA code as string (e.g., "6409")
    """
    # NACA 4-digit format: MPTT
    # M: Maximum camber in percentage of chord (first digit)
    # P: Position of maximum camber in tenths of chord (second digit)
    # TT: Maximum thickness in percentage of chord (last two digits)
    
    # For simplicity, use moderate camber (6% = digit 6)
    m = 6
    
    # Position: convert [0,1] to [0,9] for the second digit
    # 0 means leading edge, but NACA P=0 is special (symmetric), so we map [0,1] to [1,9]
    # However, location typically ranges around 0.3-0.5 for most airfoils
    # Map [0,1] to [0,9], where 0->0, 0.5->4 (40% chord), 1->9
    p = int(round(max_thickness_location * 9.0))
    
    # Thickness: ensure it's in valid range and convert to two-digit format
    # max_thickness is already a percentage (e.g., 9.0 for 9%)
    tt = int(round(max_thickness))
    tt = max(1, min(99, tt))  # Clamp to valid range
    
    return f"{m}{p}{tt:02d}"


def generate_chord_lengths(average_chord: float, chord_variance: float, n_sections: int = 6) -> List[float]:
    """
    Generate chord lengths for each section based on average and variance.
    
    Args:
        average_chord: Average chord length across all sections
        chord_variance: Variance [0,1] indicating expansion/shrink pattern
                       0 = constant chord, 1 = maximum variation (expand then shrink)
    
    Returns:
        List of chord lengths for each section
    """
    if n_sections < 2:
        return [average_chord] * n_sections
    
    # Create a pattern that expands from root, reaches maximum around middle, then shrinks to tip
    # Pattern based on sample: starts smaller, expands to peak, then shrinks to very small at tip
    chord_lengths = []
    
    # Define the shape: use a sin-based curve that peaks around 1/3 to 1/2 of span
    for i in range(n_sections):
        # Normalized position [0, 1]
        t = i / (n_sections - 1)
        
        # Create an asymmetric bell curve
        # Peak around t=0.33-0.4, then decay more gradually, steep drop at the end
        if t <= 0.4:
            # Rising phase: quadratic growth
            factor = 0.5 + 1.5 * (t / 0.4)  # From 0.5 to 2.0
        elif t <= 0.8:
            # Gradual decay
            factor = 2.0 - 1.0 * ((t - 0.4) / 0.4)  # From 2.0 to 1.0
        else:
            # Steep drop at tip
            factor = 1.0 - 0.75 * ((t - 0.8) / 0.2)  # From 1.0 to 0.25
        
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
    chord_max_thickness_location: float = 0.4,
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
        chord_max_thickness_location: Location of max thickness [0,1]
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
    naca_code = translate_to_naca_code(chord_max_thickness, chord_max_thickness_location)
    chord_lengths = generate_chord_lengths(average_chord_length, chord_length_variance, n_sections)
    twist_angles = generate_twist_angles(max_twist_angle, n_sections)
    
    # All sections use the same NACA code
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
    print(f"  NACA code: {naca_code} (thickness={chord_max_thickness}%, location={chord_max_thickness_location})")
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
  python generate_params.py output.csv
  
  # Custom overall length and number of wings
  python generate_params.py output.csv --overall-length 0.03 --n-wings 4
  
  # Custom chord thickness and location
  python generate_params.py output.csv --chord-max-thickness 12 --chord-max-thickness-location 0.3
  
  # Custom chord distribution
  python generate_params.py output.csv --average-chord-length 0.0025 --chord-length-variance 0.8
  
  # Custom twist angle
  python generate_params.py output.csv --max-twist-angle 45
        """
    )
    
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--overall-length', type=float, default=0.02,
                       help='Overall length of the wing in meters (default: 0.02)')
    parser.add_argument('--chord-max-thickness', type=float, default=9.0,
                       help='Maximum chord thickness as percentage (default: 9.0)')
    parser.add_argument('--chord-max-thickness-location', type=float, default=0.4,
                       help='Location of max thickness [0,1], 0=leading edge (default: 0.4)')
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
    if args.chord_max_thickness_location < 0 or args.chord_max_thickness_location > 1:
        parser.error("chord-max-thickness-location must be between 0 and 1")
    if args.chord_length_variance < 0 or args.chord_length_variance > 1:
        parser.error("chord-length-variance must be between 0 and 1")
    if args.n_sections < 2:
        parser.error("n-sections must be at least 2")
    
    # Generate the CSV
    generate_params_csv(
        output_file=args.output,
        overall_length=args.overall_length,
        chord_max_thickness=args.chord_max_thickness,
        chord_max_thickness_location=args.chord_max_thickness_location,
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
