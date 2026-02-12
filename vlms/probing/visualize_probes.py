"""
Probe Results Visualization Script
===================================
Generates two types of visualizations:
1. For each window, compare all probes (detection, relevance, correctness, malicious)
2. For each probe, compare all windows (decision_token, all_tokens, last_text_token, last_vision_token)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_probe_results(results_path: str) -> Dict:
    """Load probe results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_layer_auc_scores(results: Dict) -> Dict[str, Dict[str, Tuple[List[int], List[float]]]]:
    """
    Extract layer numbers and AUC scores from results.
    
    Supports two formats:
    Format 1 (converted): {window: {probe: {layers: [{layer, auc}]}}}
    Format 2 (original): {probe: {window: [{layer, auc_mean}]}}
    
    Returns:
        Dict structure: {
            'window_name': {
                'probe_name': (layer_numbers, auc_scores)
            }
        }
    """
    structured_data = {}
    
    # Auto-detect format by checking structure
    if results:
        first_key = list(results.keys())[0]
        first_value = results[first_key]
        
        # Check if this is Format 2: {probe: {window: [layers]}}
        if isinstance(first_value, dict):
            second_level = list(first_value.values())[0]
            if isinstance(second_level, list):
                # This is Format 2 - need to reorganize
                print("Detected original format: {probe: {window: [layers]}}")
                print("Reorganizing to {window: {probe: ...}}")
                
                for probe_name, windows in results.items():
                    for window_name, layer_list in windows.items():
                        if window_name not in structured_data:
                            structured_data[window_name] = {}
                        
                        layers = []
                        aucs = []
                        
                        for layer_info in layer_list:
                            if isinstance(layer_info, dict):
                                layer_num = layer_info.get('layer', None)
                                # Try different AUC field names
                                auc = layer_info.get('auc', layer_info.get('auc_mean', None))
                                
                                if layer_num is not None and auc is not None:
                                    layers.append(layer_num)
                                    aucs.append(auc)
                        
                        if layers and aucs:
                            structured_data[window_name][probe_name] = (layers, aucs)
                
                return structured_data
    
    # Format 1: {window: {probe: {layers: [{layer, auc}]}}}
    print("Detected converted format: {window: {probe: {layers: ...}}}")
    
    for window_name, window_data in results.items():
        if window_name not in structured_data:
            structured_data[window_name] = {}
        
        for probe_name, probe_data in window_data.items():
            if isinstance(probe_data, dict) and 'layers' in probe_data:
                layers = []
                aucs = []
                
                for layer_info in probe_data['layers']:
                    if isinstance(layer_info, dict):
                        layer_num = layer_info.get('layer', None)
                        auc = layer_info.get('auc', None)
                        
                        if layer_num is not None and auc is not None:
                            layers.append(layer_num)
                            aucs.append(auc)
                
                if layers and aucs:
                    structured_data[window_name][probe_name] = (layers, aucs)
    
    return structured_data

def plot_probes_per_window(data: Dict, output_dir: Path, figsize: Tuple[int, int] = (12, 8)):
    """
    Type 1: For each window, plot all probes on the same figure.
    
    Args:
        data: Structured probe data
        output_dir: Directory to save figures
        figsize: Figure size (width, height)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define nice colors and markers for different probes
    probe_styles = {
        'detection': {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5},
        'relevance': {'color': '#A23B72', 'marker': 's', 'linestyle': '-', 'linewidth': 2.5},
        'correctness': {'color': '#F18F01', 'marker': '^', 'linestyle': '-', 'linewidth': 2.5},
        'malicious': {'color': '#C73E1D', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.5},
    }
    
    for window_name, probes in data.items():
        fig, ax = plt.subplots(figsize=figsize)
        
        for probe_name, (layers, aucs) in probes.items():
            style = probe_styles.get(probe_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'linewidth': 2})
            
            ax.plot(layers, aucs, 
                   label=probe_name.capitalize(),
                   marker=style['marker'],
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   markersize=8,
                   alpha=0.8)
        
        ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Probe Comparison - {window_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
        
        # Add horizontal line at AUC = 0.5 (random baseline)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Baseline')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'window_{window_name}_all_probes.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        # Also save as PDF for publications
        output_path_pdf = output_dir / f'window_{window_name}_all_probes.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"Saved: {output_path_pdf}")
        
        plt.close()

def plot_windows_per_probe(data: Dict, output_dir: Path, figsize: Tuple[int, int] = (12, 8)):
    """
    Type 2: For each probe, plot all windows on the same figure.
    
    Args:
        data: Structured probe data
        output_dir: Directory to save figures
        figsize: Figure size (width, height)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reorganize data: probe_name -> window_name -> (layers, aucs)
    probe_centric_data = {}
    for window_name, probes in data.items():
        for probe_name, (layers, aucs) in probes.items():
            if probe_name not in probe_centric_data:
                probe_centric_data[probe_name] = {}
            probe_centric_data[probe_name][window_name] = (layers, aucs)
    
    # Define nice colors and markers for different windows
    window_styles = {
        'decision_token': {'color': '#06A77D', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5},
        'all_tokens': {'color': '#005377', 'marker': 's', 'linestyle': '-', 'linewidth': 2.5},
        'last_text_token': {'color': '#D4A574', 'marker': '^', 'linestyle': '-', 'linewidth': 2.5},
        'last_vision_token': {'color': '#8B4789', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.5},
    }
    
    for probe_name, windows in probe_centric_data.items():
        fig, ax = plt.subplots(figsize=figsize)
        
        for window_name, (layers, aucs) in windows.items():
            style = window_styles.get(window_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'linewidth': 2})
            
            ax.plot(layers, aucs,
                   label=window_name.replace('_', ' ').title(),
                   marker=style['marker'],
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   markersize=8,
                   alpha=0.8)
        
        ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Window Comparison - {probe_name.capitalize()} Probe', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
        
        # Add horizontal line at AUC = 0.5 (random baseline)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Baseline')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'probe_{probe_name}_all_windows.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        # Also save as PDF for publications
        output_path_pdf = output_dir / f'probe_{probe_name}_all_windows.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"Saved: {output_path_pdf}")
        
        plt.close()

def plot_combined_heatmap(data: Dict, output_dir: Path, figsize: Tuple[int, int] = (14, 10)):
    """
    Bonus: Create a heatmap showing AUC scores across all probes, windows, and layers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all unique layers across all probes/windows
    all_layers = set()
    for window_data in data.values():
        for layers, _ in window_data.values():
            all_layers.update(layers)
    all_layers = sorted(all_layers)
    
    # Create separate heatmap for each window
    for window_name, probes in data.items():
        probe_names = list(probes.keys())
        
        if not probe_names:
            continue
        
        # Create matrix: rows = probes, columns = layers
        matrix = np.zeros((len(probe_names), len(all_layers)))
        
        for i, probe_name in enumerate(probe_names):
            layers, aucs = probes[probe_name]
            for layer, auc in zip(layers, aucs):
                layer_idx = all_layers.index(layer)
                matrix[i, layer_idx] = auc
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(all_layers)))
        ax.set_xticklabels(all_layers, fontsize=10)
        ax.set_yticks(range(len(probe_names)))
        ax.set_yticklabels([p.capitalize() for p in probe_names], fontsize=12)
        
        ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probe Type', fontsize=14, fontweight='bold')
        ax.set_title(f'AUC Heatmap - {window_name.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('AUC Score', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(probe_names)):
            for j in range(len(all_layers)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        output_path = output_dir / f'heatmap_{window_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()

def generate_summary_statistics(data: Dict, output_dir: Path):
    """Generate summary statistics for the probe results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("PROBE RESULTS SUMMARY STATISTICS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for window_name, probes in data.items():
        summary_lines.append(f"\n{window_name.replace('_', ' ').upper()}")
        summary_lines.append("-" * 60)
        
        for probe_name, (layers, aucs) in probes.items():
            summary_lines.append(f"\n  {probe_name.capitalize()} Probe:")
            summary_lines.append(f"    Layers evaluated: {min(layers)} to {max(layers)}")
            summary_lines.append(f"    Mean AUC: {np.mean(aucs):.4f}")
            summary_lines.append(f"    Max AUC: {np.max(aucs):.4f} (Layer {layers[np.argmax(aucs)]})")
            summary_lines.append(f"    Min AUC: {np.min(aucs):.4f} (Layer {layers[np.argmin(aucs)]})")
            summary_lines.append(f"    Std Dev: {np.std(aucs):.4f}")
    
    summary_lines.append("\n" + "=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    # Print to console
    print(summary_text)
    
    # Save to file
    summary_path = output_dir / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\nSaved summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize probe results with layer-wise AUC scores')
    parser.add_argument('--results', type=str, default="./probe_analysis_llava/all_results.json",
                       help='Path to probe results JSON file')
    parser.add_argument('--output-dir', type=str, default='./visualizations_llava',
                       help='Output directory for visualizations')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                       help='Figure size (width height)')
    parser.add_argument('--skip-heatmap', action='store_true',
                       help='Skip generating heatmap visualizations')
    parser.add_argument('--windows', type=str, nargs='+', default=['last_text_token', 'last_vision_token'],
                       help='Specific windows to visualize (e.g., --windows decision_token all_tokens). If not specified, visualizes all windows.')
    parser.add_argument('--probes', type=str, nargs='+', default=None,
                       help='Specific probes to visualize (e.g., --probes detection correctness). If not specified, visualizes all probes.')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading probe results from: {args.results}")
    results = load_probe_results(args.results)
    
    # Extract structured data
    print("Extracting layer-wise AUC scores...")
    data = extract_layer_auc_scores(results)
    
    # Filter windows if specified
    if args.windows:
        print(f"\nFiltering windows: {', '.join(args.windows)}")
        filtered_data = {}
        for window in args.windows:
            if window in data:
                filtered_data[window] = data[window]
            else:
                print(f"  Warning: Window '{window}' not found in data")
        data = filtered_data
        
        if not data:
            print("Error: No valid windows selected. Available windows:", list(data.keys()))
            return
    
    # Filter probes if specified
    if args.probes:
        print(f"Filtering probes: {', '.join(args.probes)}")
        for window in data:
            filtered_probes = {}
            for probe in args.probes:
                if probe in data[window]:
                    filtered_probes[probe] = data[window][probe]
                else:
                    print(f"  Warning: Probe '{probe}' not found in window '{window}'")
            data[window] = filtered_probes
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\n1. Generating window-based comparisons (all probes per window)...")
    plot_probes_per_window(data, output_dir / 'type1_probes_per_window', 
                          figsize=tuple(args.figsize))
    
    print("\n2. Generating probe-based comparisons (all windows per probe)...")
    plot_windows_per_probe(data, output_dir / 'type2_windows_per_probe',
                          figsize=tuple(args.figsize))
    
    if not args.skip_heatmap:
        print("\n3. Generating heatmap visualizations...")
        plot_combined_heatmap(data, output_dir / 'bonus_heatmaps',
                            figsize=tuple(args.figsize))
    
    # Generate summary statistics
    print("\n4. Generating summary statistics...")
    generate_summary_statistics(data, output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nGenerated visualizations:")
    print(f"  - Type 1 (probes per window): {output_dir / 'type1_probes_per_window'}")
    print(f"  - Type 2 (windows per probe): {output_dir / 'type2_windows_per_probe'}")
    if not args.skip_heatmap:
        print(f"  - Bonus heatmaps: {output_dir / 'bonus_heatmaps'}")
    print(f"  - Summary statistics: {output_dir / 'summary_statistics.txt'}")

if __name__ == '__main__':
    main()