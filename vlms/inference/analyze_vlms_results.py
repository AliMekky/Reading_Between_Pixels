#!/usr/bin/env python3
"""
Analyze VLM Inference Results

This script:
1. Calculates accuracy for each result file
2. Analyzes performance by category (Instance Attributes, Instances Counting, etc.)
3. Analyzes performance by quality level (High, Medium, Low)
4. Generates comparison tables and visualizations
5. Creates summary reports

Usage:
    python analyze_vlm_results.py --results_dir ./results --questions_dir ./filtered_data --output ./analysis
"""

import json
import os
import argparse
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

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
    if not filename.endswith('_results.json'):
        return None, None
    
    # Remove '_results.json'
    base = filename.replace('_results.json', '')
    
    # Split by last underscore to get variant
    parts = base.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    
    return None, None

def calculate_accuracy(results, questions_data):
    """
    Calculate accuracy by matching results with original questions
    
    Args:
        results: List of result dicts with 'question_id' and 'predicted_answer'
        questions_data: List of question dicts with 'question_id' and 'answer'
    
    Returns:
        dict with accuracy metrics
    """
    # Create lookup dict for questions
    questions_dict = {q['image']: q for q in questions_data}
    
    total = 0
    correct = 0
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_quality = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    incorrect_samples = []
    
    for result in results:
        qid = result.get('image')
        predicted = result.get('predicted_answer', '').strip().upper()
        
        if qid not in questions_dict:
            print(f"Warning: Question ID {qid} not found in questions data")
            continue
        
        question = questions_dict[qid]
        ground_truth = question.get('answer', '').strip().upper()
        category = question.get('category', 'Unknown')
        quality = question.get('quality', 'Unknown')
        
        total += 1
        by_category[category]['total'] += 1
        by_quality[quality]['total'] += 1
        
        if predicted == ground_truth:
            correct += 1
            by_category[category]['correct'] += 1
            by_quality[quality]['correct'] += 1
        else:
            incorrect_samples.append({
                'question_id': qid,
                'question': question.get('question', ''),
                'ground_truth': ground_truth,
                'predicted': predicted,
                'category': category,
                'quality': quality,
                'image': question.get('image', '')
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
    
    quality_acc = {}
    for qual, stats in by_quality.items():
        quality_acc[qual] = {
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
        'by_quality': quality_acc,
        'incorrect_samples': incorrect_samples[:20]  # Keep first 20 for analysis
    }

def analyze_all_results(results_dir, questions_dir):
    """Analyze all result files"""
    
    all_analyses = {}
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    
    print(f"Found {len(result_files)} result files")
    print()
    
    for result_file in sorted(result_files):
        print(f"Analyzing: {result_file}")
        
        # Parse filename
        model, variant = parse_filename(result_file)
        if not model or not variant:
            print(f"  ⚠️  Could not parse filename: {result_file}")
            continue
        
        # Load results
        results_path = os.path.join(results_dir, result_file)
        results = load_json(results_path)
        if not results:
            print(f"  ❌ Failed to load results")
            continue
        
        # Load corresponding questions
        questions_file = f"filtered_questions_{variant}.json"
        questions_path = os.path.join(questions_dir, questions_file)
        questions = load_json(questions_path)
        if not questions:
            print(f"  ❌ Failed to load questions from {questions_file}")
            continue
        
        # Calculate accuracy
        analysis = calculate_accuracy(results, questions)
        
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
    
    # Quality accuracy table
    quality_data = []
    for model, variants in all_analyses.items():
        for variant, analysis in variants.items():
            for quality, stats in analysis['by_quality'].items():
                quality_data.append({
                    'Model': model,
                    'Variant': variant,
                    'Quality': quality,
                    'Accuracy (%)': round(stats['accuracy'], 2),
                    'Correct': stats['correct'],
                    'Total': stats['total']
                })
    
    quality_df = pd.DataFrame(quality_data)
    
    return overall_df, category_df, quality_df

def plot_overall_accuracy(overall_df, output_dir):
    """Plot overall accuracy comparison"""
    
    # Pivot for grouped bar chart
    pivot_df = overall_df.pivot(index='Model', columns='Variant', values='Accuracy (%)')
    
    plt.figure(figsize=(14, 6))
    ax = pivot_df.plot(kind='bar', width=0.8)
    plt.title('Overall Accuracy by Model and Variant', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(title='Variant', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'overall_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_category_accuracy(category_df, output_dir):
    """Plot accuracy by category"""
    
    categories = category_df['Category'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, category in enumerate(sorted(categories)):
        if idx >= len(axes):
            break
        
        cat_data = category_df[category_df['Category'] == category]
        pivot = cat_data.pivot(index='Model', columns='Variant', values='Accuracy (%)')
        
        pivot.plot(kind='bar', ax=axes[idx], width=0.8)
        axes[idx].set_title(f'{category}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Model', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        axes[idx].legend(title='Variant', fontsize=8)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Accuracy by Category', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'category_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_quality_accuracy(quality_df, output_dir):
    """Plot accuracy by quality level"""
    
    plt.figure(figsize=(14, 6))
    
    # Create grouped bar chart
    quality_order = ['High', 'Medium', 'Low']
    models = quality_df['Model'].unique()
    
    for model in sorted(models):
        model_data = quality_df[quality_df['Model'] == model]
        
        for variant in sorted(model_data['Variant'].unique()):
            variant_data = model_data[model_data['Variant'] == variant]
            
            # Reorder by quality
            variant_data = variant_data.set_index('Quality').reindex(quality_order).reset_index()
            
            label = f"{model}_{variant}"
            plt.plot(variant_data['Quality'], variant_data['Accuracy (%)'], 
                    marker='o', label=label, linewidth=2, markersize=8)
    
    plt.title('Accuracy by Quality Level', fontsize=16, fontweight='bold')
    plt.xlabel('Quality Level', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'quality_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_heatmap_comparison(overall_df, output_dir):
    """Create heatmap of model vs variant accuracy"""
    
    pivot = overall_df.pivot(index='Model', columns='Variant', values='Accuracy (%)')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=100, linewidths=0.5, cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Accuracy Heatmap: Model vs Variant', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Variant', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'accuracy_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_baseline_comparison(overall_df, output_dir):
    """
    Create grouped barplot with no_text as baseline followed by all variants
    All models shown together with horizontal lines extending from no_text bars
    """
    
    # Define variant order
    variant_order = ['notext', 'correct', 'relevant', 'irrelevant', 'misleading']
    
    # Check if notext variant exists
    if 'notext' not in overall_df['Variant'].values:
        print("Warning: no_text variant not found, skipping baseline plot")
        return
    
    # Filter to only include variants in our order
    plot_df = overall_df[overall_df['Variant'].isin(variant_order)].copy()
    
    # Set categorical order
    plot_df['Variant'] = pd.Categorical(plot_df['Variant'], 
                                        categories=variant_order, 
                                        ordered=True)
    plot_df = plot_df.sort_values(['Variant', 'Model'])
    
    # Get models
    models = sorted(plot_df['Model'].unique())
    variants = variant_order
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define colors for variants
    variant_colors = {
        'no_text': '#3498db',      # Blue
        'correct': '#2ecc71',      # Green
        'relevant': '#f39c12',     # Orange
        'irrelevant': '#e74c3c',   # Red
        'misleading': '#9b59b6'    # Purple
    }
    
    # Calculate bar positions
    n_models = len(models)
    n_variants = len(variants)
    bar_width = 0.15
    group_width = bar_width * n_models
    group_gap = 0.3
    
    # Calculate x positions for each group (variant)
    group_positions = []
    current_pos = 0
    for i in range(n_variants):
        group_positions.append(current_pos)
        current_pos += group_width + group_gap
    
    # Plot bars for each model and store information
    model_bars = {}
    baseline_info = {}  # Store no_text bar position and accuracy for each model
    
    for model_idx, model in enumerate(models):
        model_data = plot_df[plot_df['Model'] == model]
        
        x_positions = []
        accuracies = []
        colors = []
        
        for variant_idx, variant in enumerate(variants):
            variant_data = model_data[model_data['Variant'] == variant]
            
            if len(variant_data) > 0:
                x_pos = group_positions[variant_idx] + (model_idx * bar_width)
                acc = variant_data['Accuracy (%)'].values[0]
                
                x_positions.append(x_pos)
                accuracies.append(acc)
                colors.append(variant_colors.get(variant, '#95a5a6'))
                
                # Store baseline info
                if variant == 'no_text':
                    baseline_info[model] = {
                        'x': x_pos,
                        'accuracy': acc,
                        'color': variant_colors['no_text']
                    }
            else:
                # Variant doesn't exist for this model
                x_pos = group_positions[variant_idx] + (model_idx * bar_width)
                x_positions.append(x_pos)
                accuracies.append(0)
                colors.append('#cccccc')
        
        bars = ax.bar(x_positions, accuracies, bar_width, 
                     label=model, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Store bars for legend
        model_bars[model] = bars[0]
        
        # Add percentage labels on bars
        for x, acc, bar in zip(x_positions, accuracies, bars):
            if acc > 0:  # Only label if there's data
                ax.text(x, acc + 1.5, f'{acc:.1f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Draw horizontal lines from no_text baseline extending across all variants
    for model, info in baseline_info.items():
        baseline_acc = info['accuracy']
        baseline_x = info['x']
        
        # Get the x position of the last variant for this model
        last_variant_x = group_positions[-1] + (models.index(model) * bar_width)
        
        # Draw horizontal dashed line from no_text bar to the end
        # Use higher zorder to ensure it's on top of bars
        ax.plot([baseline_x + bar_width, last_variant_x + bar_width], 
               [baseline_acc, baseline_acc],
               linestyle='--', linewidth=3, alpha=0.8,
               color='black', zorder=100)  # Changed to black and higher zorder
        
        # Add label at the end of the line (right side)
        ax.text(last_variant_x + bar_width + 0.1, baseline_acc, 
               f'{info["accuracy"]:.1f}% ({model})',
               ha='left', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='black', alpha=0.9, linewidth=1.5))
    
    # Set x-axis labels at group centers
    group_centers = [pos + (group_width / 2) - (bar_width / 2) for pos in group_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(variants, fontsize=12, fontweight='bold')
    
    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('All Models - Accuracy Comparison with No-Text Baseline\n(Horizontal dashed lines from no-text baseline)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(title='Model', loc='upper right', fontsize=10, title_fontsize=11, framealpha=0.9)
    
    # Add variant color legend on the side
    from matplotlib.patches import Patch
    variant_patches = [Patch(facecolor=variant_colors[v], label=v, alpha=0.8) 
                      for v in variants if v in variant_colors]
    
    # Create second legend for variants
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=variant_patches, title='Variant Type', 
              loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'all_models_baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Also create a version with difference from baseline annotated
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Reset baseline_info
    baseline_info = {}
    
    # Plot bars again
    for model_idx, model in enumerate(models):
        model_data = plot_df[plot_df['Model'] == model]
        baseline_data = model_data[model_data['Variant'] == 'no_text']
        
        if len(baseline_data) == 0:
            continue
            
        baseline_acc = baseline_data['Accuracy (%)'].values[0]
        
        x_positions = []
        accuracies = []
        colors = []
        
        for variant_idx, variant in enumerate(variants):
            variant_data = model_data[model_data['Variant'] == variant]
            
            if len(variant_data) > 0:
                x_pos = group_positions[variant_idx] + (model_idx * bar_width)
                acc = variant_data['Accuracy (%)'].values[0]
                x_positions.append(x_pos)
                accuracies.append(acc)
                colors.append(variant_colors.get(variant, '#95a5a6'))
                
                if variant == 'no_text':
                    baseline_info[model] = {
                        'x': x_pos,
                        'accuracy': acc,
                        'color': variant_colors['no_text']
                    }
            else:
                x_pos = group_positions[variant_idx] + (model_idx * bar_width)
                x_positions.append(x_pos)
                accuracies.append(0)
                colors.append('#cccccc')
        
        bars = ax.bar(x_positions, accuracies, bar_width, 
                     label=model, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels and difference from baseline
        for idx, (x, acc, bar) in enumerate(zip(x_positions, accuracies, bars)):
            if acc > 0:
                # Add percentage on top
                ax.text(x, acc + 1.5, f'{acc:.1f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # Add difference for non-baseline variants
                if idx > 0 and acc > 0:  # Skip no_text itself
                    diff = acc - baseline_acc
                    diff_color = '#2ecc71' if diff >= 0 else '#e74c3c'
                    ax.text(x, acc/2, f'{diff:+.1f}%',
                           ha='center', va='center', fontsize=7,
                           fontweight='bold', color=diff_color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor=diff_color, alpha=0.9, linewidth=1.5))
    
    # Draw horizontal lines from no_text baseline
    for model, info in baseline_info.items():
        baseline_acc = info['accuracy']
        baseline_x = info['x']
        last_variant_x = group_positions[-1] + (models.index(model) * bar_width)
        
        # Draw horizontal dashed line from after no_text bar to the end
        ax.plot([baseline_x + bar_width, last_variant_x + bar_width], 
               [baseline_acc, baseline_acc],
               linestyle='--', linewidth=3, alpha=0.8,
               color='black', zorder=100)  # Changed to black and higher zorder
        
        # Add label at the end of the line
        ax.text(last_variant_x + bar_width + 0.1, baseline_acc, 
               f'{info["accuracy"]:.1f}% ({model})',
               ha='left', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='black', alpha=0.9, linewidth=1.5))
    
    group_centers = [pos + (group_width / 2) - (bar_width / 2) for pos in group_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(variants, fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('All Models - Accuracy with Differences from No-Text Baseline\n(Horizontal lines from baseline, boxes show % difference)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(title='Model', loc='upper right', fontsize=10, title_fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'all_models_baseline_with_diff.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_text_report(all_analyses, overall_df, category_df, quality_df, output_dir):
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
        
        # Quality performance
        f.write("\n" + "=" * 80 + "\n")
        f.write("ACCURACY BY QUALITY LEVEL\n")
        f.write("=" * 80 + "\n\n")
        
        for quality in ['High', 'Medium', 'Low']:
            if quality in quality_df['Quality'].values:
                f.write(f"\n{quality} Quality\n")
                f.write("-" * 80 + "\n")
                qual_data = quality_df[quality_df['Quality'] == quality][['Model', 'Variant', 'Accuracy (%)', 'Correct', 'Total']]
                f.write(qual_data.to_string(index=False))
                f.write("\n")
        
        # Key insights
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Variant comparison
        f.write("1. VARIANT COMPARISON:\n")
        for variant in ['correct', 'relevant', 'irrelevant', 'misleading']:
            if variant in variant_avg.index:
                f.write(f"   - {variant:12s}: {variant_avg[variant]:.2f}% average accuracy\n")
        
        f.write("\n2. CATEGORY INSIGHTS:\n")
        cat_avg = category_df.groupby('Category')['Accuracy (%)'].mean().sort_values(ascending=False)
        for cat, acc in cat_avg.items():
            f.write(f"   - {cat:30s}: {acc:.2f}% average\n")
        
        f.write("\n3. QUALITY INSIGHTS:\n")
        qual_avg = quality_df.groupby('Quality')['Accuracy (%)'].mean().sort_values(ascending=False)
        for qual, acc in qual_avg.items():
            f.write(f"   - {qual:10s} quality: {acc:.2f}% average\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze VLM inference results')
    parser.add_argument('--results_dir', '-r', default='./results',
                       help='Directory containing result JSON files')
    parser.add_argument('--questions_dir', '-q', default='./filtered_data',
                       help='Directory containing original question JSON files')
    parser.add_argument('--output', '-o', default='./analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
    
    print("=" * 80)
    print("VLM RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze all results
    print("📊 Analyzing results...")
    all_analyses = analyze_all_results(args.results_dir, args.questions_dir)
    
    if not all_analyses:
        print("❌ No analyses completed")
        return
    
    print()
    print("📈 Creating summary tables...")
    overall_df, category_df, quality_df = create_summary_tables(all_analyses)
    
    # Save tables to CSV
    overall_df.to_csv(os.path.join(args.output, 'overall_accuracy.csv'), index=False)
    category_df.to_csv(os.path.join(args.output, 'category_accuracy.csv'), index=False)
    quality_df.to_csv(os.path.join(args.output, 'quality_accuracy.csv'), index=False)
    print(f"✅ Saved CSV tables to {args.output}")
    
    print()
    print("📊 Generating visualizations...")
    plots_dir = os.path.join(args.output, 'plots')
    
    plot_overall_accuracy(overall_df, plots_dir)
    plot_category_accuracy(category_df, plots_dir)
    plot_quality_accuracy(quality_df, plots_dir)
    plot_heatmap_comparison(overall_df, plots_dir)
    plot_baseline_comparison(overall_df, plots_dir)
    
    print()
    print("📝 Generating text report...")
    generate_text_report(all_analyses, overall_df, category_df, quality_df, args.output)
    
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
    print("   - quality_accuracy.csv")
    print("   - analysis_report.txt")
    print("   - detailed_analysis.json")
    print("   - plots/overall_accuracy.png")
    print("   - plots/category_accuracy.png")
    print("   - plots/quality_accuracy.png")
    print("   - plots/accuracy_heatmap.png")
    print("   - plots/all_models_baseline_comparison.png")
    print("   - plots/all_models_baseline_with_diff.png")

if __name__ == "__main__":
    main()