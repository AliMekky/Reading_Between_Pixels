#!/usr/bin/env python3
"""
Analyze VLM Inference Results - HuggingFace Dataset Version

This script:
1. Loads results and questions from HuggingFace dataset
2. Calculates accuracy for each result file
3. Analyzes performance by category
4. Generates comparison tables and visualizations
5. Creates summary reports

Usage:
    python analyze_vlm_results.py --results_dir ./results --hf_dataset AHAAM/CIM --output ./analysis
"""

import json
import os
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def sanitize_repo_id(repo_id: str) -> str:
    """Make a filesystem-safe name for caching."""
    return repo_id.replace("/", "__").replace(" ", "_")

def get_or_download_hf_dataset(
    dataset_id: str, 
    local_cache_root: str = "./hf_dataset_local_cache",
    split: str = "test"
):
    """Download or load cached HF dataset."""
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_repo_id(dataset_id)
    cache_dir = local_cache_root / safe_name

    if cache_dir.exists():
        print(f"✓ Loading dataset from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"⬇️  Downloading '{dataset_id}'...")
    ds = load_dataset(dataset_id, split=split)
    
    try:
        ds.save_to_disk(str(cache_dir))
        print(f"✓ Saved to cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️  Cache save failed: {e}")
    
    return ds

def load_hf_dataset_as_dict(hf_dataset):
    """
    Convert HF dataset to a dict keyed by question_id for fast lookup.
    
    Returns:
        dict: {question_id: {question, choices, answer, category, etc.}}
    """
    questions_dict = {}
    
    if isinstance(hf_dataset, DatasetDict):
        split_name = "test" if "test" in hf_dataset else list(hf_dataset.keys())[0]
        dataset = hf_dataset[split_name]
    else:
        dataset = hf_dataset
    
    print(f"Loading {len(dataset)} questions from HF dataset...")
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            question_id = sample.get("question_id") or sample.get("image_id") or f"q_{idx}"
            
            questions_dict[question_id] = {
                'question_id': question_id,
                'question': sample.get('question', ''),
                'choices': sample.get('choices', []),
                'answer': sample.get('answer', ''),
                'category': sample.get('category', 'Unknown'),
                'image_id': sample.get('image_id', ''),
            }
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            continue
    
    print(f"✓ Loaded {len(questions_dict)} questions")
    return questions_dict

def load_json(filepath):
    """Load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def parse_filename(filename):
    """
    Parse result filename to extract model and variant
    Expected format: {model}_{variant}_results.json
    Returns: (model, variant) or (None, None)
    """
    
    
    parts = filename.split('_')
    if len(parts) == 3:
        return parts[0], parts[2].split('.')[0]
    
    return None, None

def calculate_accuracy(results, questions_dict):
    """
    Calculate accuracy by matching results with original questions
    
    Args:
        results: List of result dicts with 'image_id' and 'predicted_answer'
        questions_dict: Dict of question data keyed by question_id
    
    Returns:
        dict with accuracy metrics
    """
    total = 0
    correct = 0
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    incorrect_samples = []
    
    for result in results:
        # Try multiple possible ID fields
        qid = result.get('image_id') or result.get('question_id') or result.get('image')
        predicted = result.get('predicted_answer', '').strip().upper()
        
        if qid not in questions_dict:
            print(f"⚠️  Question ID {qid} not found in questions data")
            continue
        
        question = questions_dict[qid]
        ground_truth = question.get('answer', '').strip().upper()
        category = question.get('category', 'Unknown')
        
        total += 1
        by_category[category]['total'] += 1
        
        if predicted == ground_truth:
            correct += 1
            by_category[category]['correct'] += 1
        else:
            incorrect_samples.append({
                'question_id': qid,
                'question': question.get('question', ''),
                'ground_truth': ground_truth,
                'predicted': predicted,
                'category': category,
            })
    
    # Calculate percentages
    overall_acc = (correct / total * 100) if total > 0 else 0
    
    category_acc = {}
    for cat, stats in by_category.items():
        category_acc[cat] = {
            'correct': stats['correct'],
            'total': stats['total'],
            'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        }
    
    return {
        'overall': {
            'correct': correct,
            'total': total,
            'accuracy': overall_acc
        },
        'by_category': category_acc,
        'incorrect_samples': incorrect_samples[:20]  # Keep first 20 for analysis
    }

def analyze_all_results(results_dir, questions_dict):
    """Analyze all result files using HF dataset questions"""
    
    all_analyses = {}
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    print(f"\nFound {len(result_files)} result files")
    print()
    
    for result_file in sorted(result_files):
        print(f"Analyzing: {result_file}")
        
        # Parse filename
        model, variant = parse_filename(result_file)
        if variant == 'summary':
            continue
        print(f"  Model: {model}, Variant: {variant}")
        if not model or not variant:
            print(f"  ⚠️  Could not parse filename: {result_file}")
            continue
        
        # Load results
        results_path = os.path.join(results_dir, result_file)
        results = load_json(results_path)
        if not results:
            print(f"  ❌ Failed to load results")
            continue
        
        # Calculate accuracy
        analysis = calculate_accuracy(results, questions_dict)
        
        print(f"  ✅ Overall Accuracy: {analysis['overall']['accuracy']:.2f}% ({analysis['overall']['correct']}/{analysis['overall']['total']})")
        
        # Store analysis
        if model not in all_analyses:
            all_analyses[model] = {}
        all_analyses[model][variant] = analysis
        print()
    
    return all_analyses

def create_summary_tables(all_analyses):
    """Create summary tables as DataFrames"""
    
    # Overall accuracy table
    overall_data = []
    for model, variants in all_analyses.items():
        for variant, analysis in variants.items():
            overall_data.append({
                'Model': model,
                'Variant': variant,
                'Accuracy (%)': round(analysis['overall']['accuracy'], 2),
                'Correct': analysis['overall']['correct'],
                'Total': analysis['overall']['total']
            })
    
    overall_df = pd.DataFrame(overall_data)
    
    # Category accuracy table
    category_data = []
    for model, variants in all_analyses.items():
        for variant, analysis in variants.items():
            for category, stats in analysis['by_category'].items():
                category_data.append({
                    'Model': model,
                    'Variant': variant,
                    'Category': category,
                    'Accuracy (%)': round(stats['accuracy'], 2),
                    'Correct': stats['correct'],
                    'Total': stats['total']
                })
    
    category_df = pd.DataFrame(category_data)
    
    return overall_df, category_df

def plot_category_baseline_comparison(category_df, overall_df, output_dir):
    """
    Create per-category plots with notext baseline AS A BAR
    Shows difference from baseline inside bars with bar-width boxes
    """
    categories = sorted(category_df['Category'].unique())
    
    # Define variant order and colors
    variant_order = ['correct', 'irrelevant', 'notext', 'misleading']
    variant_colors = {
        'notext': '#95a5a6',       # Grey (baseline)
        'correct': '#27ae60',      # Forest Green
        'irrelevant': '#f39c12',   # Orange
        'misleading': '#e74c3c'    # Red
    }
    
    # Determine grid size based on number of categories
    n_categories = len(categories)
    if n_categories <= 2:
        n_rows, n_cols = 1, 2
        figsize = (20, 8)
    elif n_categories <= 4:
        n_rows, n_cols = 2, 2
        figsize = (20, 16)
    else:
        n_rows = (n_categories + 1) // 2
        n_cols = 2
        figsize = (20, 8 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_categories == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if n_categories > 2 else axes
    
    for idx, category in enumerate(categories):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get data for this category
        cat_data = category_df[category_df['Category'] == category].copy()
        
        # Filter to all variants including notext
        cat_data = cat_data[cat_data['Variant'].isin(variant_order)]
        
        if len(cat_data) == 0:
            ax.set_visible(False)
            continue
        
        # Set categorical order
        cat_data['Variant'] = pd.Categorical(cat_data['Variant'], 
                                            categories=variant_order, 
                                            ordered=True)
        cat_data = cat_data.sort_values(['Model', 'Variant'])
        
        models = sorted(cat_data['Model'].unique())
        
        # Calculate bar positions
        n_models = len(models)
        n_variants = len(variant_order)
        bar_width = 0.5
        group_gap = 0.5
        
        # x positions for each model group
        model_positions = [i * (n_variants * bar_width + group_gap) for i in range(n_models)]
        
        # Plot bars for each model
        for model_idx, model in enumerate(models):
            model_data = cat_data[cat_data['Model'] == model]
            base_x = model_positions[model_idx]
            
            # Get notext baseline value for this model
            notext_data = model_data[model_data['Variant'] == 'notext']
            notext_acc = notext_data['Accuracy (%)'].values[0] if len(notext_data) > 0 else None
            
            # Plot each variant
            for variant_idx, variant in enumerate(variant_order):
                variant_data = model_data[model_data['Variant'] == variant]
                
                if len(variant_data) > 0:
                    x_pos = base_x + (variant_idx * bar_width)
                    acc = variant_data['Accuracy (%)'].values[0]
                    
                    # Plot bar
                    bar = ax.bar(x_pos, acc, bar_width,
                               color=variant_colors[variant],
                               alpha=0.85,
                               edgecolor='black',
                               linewidth=1.5,
                               zorder=3)
                    
                    # Add percentage label on top of bar
                    ax.text(x_pos, acc + 2, f'{acc:.1f}%',
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold',
                           zorder=6)
                    
                    # Add difference from baseline (if not baseline itself)
                    if variant != 'notext' and notext_acc is not None:
                        diff = acc - notext_acc
                        
                        # Choose symbol based on difference
                        if diff > 0:
                            symbol = '↑'
                            diff_text = f'{symbol} {diff:.1f}%'
                        elif diff < 0:
                            symbol = '↓'
                            diff_text = f'{symbol} {abs(diff):.1f}%'
                        else:
                            symbol = '='
                            diff_text = f'{symbol} 0.0%'
                        
                        # Position based on bar height
                        if acc > 30:
                            # Inside bar, middle position
                            y_pos = acc * 0.5
                            text_color = 'white'
                            bg_color = 'black'
                            bg_alpha = 0.75
                        elif acc > 15:
                            # Inside bar, lower position
                            y_pos = acc * 0.35
                            text_color = 'white'
                            bg_color = 'black'
                            bg_alpha = 0.75
                        else:
                            # Very short bar - place just above x-axis
                            y_pos = 7
                            text_color = 'black'
                            bg_color = 'white'
                            bg_alpha = 0.95
                        
                        # Create a rectangle patch for the box (same width as bar)
                        from matplotlib.patches import FancyBboxPatch
                        
                        # Calculate box dimensions
                        box_height = 6  # Fixed height
                        box_y = y_pos - box_height/2
                        
                        # Draw the box
                        # bbox = FancyBboxPatch(
                        #     (x_pos - bar_width/2, box_y),
                        #     bar_width, box_height,
                        #     boxstyle="round,pad=0.02",
                        #     facecolor=bg_color,
                        #     edgecolor='white' if text_color == 'white' else 'black',
                        #     linewidth=1.5,
                        #     alpha=bg_alpha,
                        #     zorder=5
                        # )
                        # ax.add_patch(bbox)
                        
                        # Add the difference text
                        ax.text(x_pos, y_pos, 
                               diff_text,
                               ha='center', va='center',
                               fontsize=9, fontweight='bold',
                               color=text_color,
                               zorder=6)
        
        # Set x-axis labels (model names centered under each group)
        model_centers = [pos + (n_variants * bar_width) / 2 - bar_width / 2 
                        for pos in model_positions]
        ax.set_xticks(model_centers)
        ax.set_xticklabels(models, fontsize=13, fontweight='bold')
        
        # Labels and styling
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(category, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 115)
        
        # Enhanced grid
        ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1, zorder=1)
        ax.set_axisbelow(True)
        
        # Add legend (only on first subplot)
        if idx == 0:
            from matplotlib.patches import Patch
            
            legend_elements = [
                Patch(facecolor=variant_colors['notext'], label='no-text (baseline)', 
                     edgecolor='black', linewidth=1.5, alpha=0.85),
                Patch(facecolor=variant_colors['correct'], label='correct', 
                     edgecolor='black', linewidth=1.5, alpha=0.85),
                Patch(facecolor=variant_colors['irrelevant'], label='irrelevant', 
                     edgecolor='black', linewidth=1.5, alpha=0.85),
                Patch(facecolor=variant_colors['misleading'], label='misleading', 
                     edgecolor='black', linewidth=1.5, alpha=0.85),
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                     fontsize=11, framealpha=0.95, edgecolor='black', 
                     fancybox=True, shadow=True)
    
    # Hide unused subplots
    for idx in range(len(categories), len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle('Accuracy by Category (with No-Text Baseline)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'category_baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_text_report(all_analyses, overall_df, category_df, output_dir):
    """Generate comprehensive text report"""
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VLM INFERENCE RESULTS - COMPREHENSIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Summary
        f.write("OVERALL ACCURACY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(overall_df.to_string(index=False))
        f.write("\n\n")
        
        # Best and Worst performers
        f.write("PERFORMANCE HIGHLIGHTS\n")
        f.write("-" * 80 + "\n")
        best_row = overall_df.loc[overall_df['Accuracy (%)'].idxmax()]
        worst_row = overall_df.loc[overall_df['Accuracy (%)'].idxmin()]
        f.write(f"🏆 Best Performance: {best_row['Model']} on {best_row['Variant']} - {best_row['Accuracy (%)']}%\n")
        f.write(f"⚠️  Worst Performance: {worst_row['Model']} on {worst_row['Variant']} - {worst_row['Accuracy (%)']}%\n")
        f.write("\n")
        
        # Average by model
        f.write("AVERAGE ACCURACY BY MODEL\n")
        f.write("-" * 80 + "\n")
        model_avg = overall_df.groupby('Model')['Accuracy (%)'].mean().sort_values(ascending=False)
        for model, acc in model_avg.items():
            f.write(f"  {model:20s}: {acc:.2f}%\n")
        f.write("\n")
        
        # Average by variant
        f.write("AVERAGE ACCURACY BY VARIANT\n")
        f.write("-" * 80 + "\n")
        variant_avg = overall_df.groupby('Variant')['Accuracy (%)'].mean().sort_values(ascending=False)
        for variant, acc in variant_avg.items():
            f.write(f"  {variant:20s}: {acc:.2f}%\n")
        f.write("\n")
        
        # Category performance
        f.write("\n" + "=" * 80 + "\n")
        f.write("ACCURACY BY CATEGORY\n")
        f.write("=" * 80 + "\n\n")
        
        for category in sorted(category_df['Category'].unique()):
            f.write(f"\n{category}\n")
            f.write("-" * 80 + "\n")
            cat_data = category_df[category_df['Category'] == category][['Model', 'Variant', 'Accuracy (%)', 'Correct', 'Total']]
            f.write(cat_data.to_string(index=False))
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze VLM inference results from HF dataset')
    parser.add_argument('--results_dir', '-r', default='./results',
                       help='Directory containing result JSON files')
    parser.add_argument('--hf_dataset', default="AHAAM/CIM",
                       help='HuggingFace dataset ID (e.g., AHAAM/CIM)')
    parser.add_argument('--hf_cache_dir', default='./hf_cache/AHAAM__CIM/AHAAM__CIM',
                       help='Local cache directory for HF dataset')
    parser.add_argument('--output', '-o', default='./benchmarking',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
    
    print("=" * 80)
    print("VLM RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Load HF dataset
    print("📥 Loading HuggingFace dataset...")
    hf_dataset = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split="test")
    questions_dict = load_hf_dataset_as_dict(hf_dataset)
    
    print()
    print("📊 Analyzing results...")
    all_analyses = analyze_all_results(args.results_dir, questions_dict)
    
    if not all_analyses:
        print("❌ No analyses completed")
        return
    
    print()
    print("📈 Creating summary tables...")
    overall_df, category_df = create_summary_tables(all_analyses)
    
    # Save tables to CSV
    overall_df.to_csv(os.path.join(args.output, 'overall_accuracy.csv'), index=False)
    category_df.to_csv(os.path.join(args.output, 'category_accuracy.csv'), index=False)
    print(f"✅ Saved CSV tables to {args.output}")
    
    print()
    print("📊 Generating visualizations...")
    plots_dir = os.path.join(args.output, 'plots')
    
    plot_category_baseline_comparison(category_df, overall_df, plots_dir)
    
    print()
    print("📝 Generating text report...")
    generate_text_report(all_analyses, overall_df, category_df, args.output)
    
    # Save detailed analysis as JSON
    analysis_json_path = os.path.join(args.output, 'detailed_analysis.json')
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2)
    print(f"Saved: {analysis_json_path}")
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"📂 Output directory: {args.output}")
    print("📋 Generated files:")
    print("   - overall_accuracy.csv")
    print("   - category_accuracy.csv")
    print("   - analysis_report.txt")
    print("   - detailed_analysis.json")
    print("   - plots/overall_accuracy.png")
    print("   - plots/category_baseline_comparison.png")
    print("   - plots/accuracy_heatmap.png")
    print("   - plots/all_models_baseline_comparison.png")

if __name__ == "__main__":
    main()