#!/usr/bin/env python3
"""
Linear Probe Analysis for VLM Text Processing

Clean implementation of 3 essential probes:
- Detection: N vs {C, M, I} - Does the model detect text presence?
- Relevance: {C, M} vs I - Does the model distinguish relevant from irrelevant text?
- Correctness: C vs M - Does the model distinguish correct from misleading text?

Each probe tested across 4 token windows and 32 layers.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Simple configuration."""
    ACTIVATIONS_DIR = "../activations_analysis/llava_next_data_v2_activations_new_windows"
    OUTPUT_DIR = "./probe_analysis_llava_next"
    WINDOWS = ['single_token', 'last_vision_token', 'all_tokens']

    NUM_LAYERS = 32
    CV_FOLDS = 5
    RANDOM_SEED = 42


# ============================================================================
# Data Loading
# ============================================================================

def load_all_activations(activations_dir):
    """
    Load all activation files.
    
    Returns:
        List of dicts with activation data
    """
    activations_dir = Path(activations_dir)
    files = sorted(activations_dir.glob("*.pt"))
    
    print(f"Loading {len(files)} activation files...")
    
    data = []
    for file_path in tqdm(files, desc="Loading"):
        tensors = torch.load(file_path, map_location='cpu')
        data.append(tensors)
    
    print(f"✓ Loaded {len(data)} questions\n")
    return data


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features_detection(data, layer_idx, window):
    """
    Extract detection features: N vs {C, M, I}
    
    Tests: Does the model detect text presence at all?
    
    Returns:
        X: [1416, 4096] - features (354 no-text + 1062 text)
        y: [1416] - labels (0=no text, 1=has text)
        groups: [1416] - question IDs
    """
    X_notext = []
    X_text = []
    groups = []
    
    for q_id, tensors in enumerate(data):
        N = tensors[f'notext_hidden_states_{window}'][layer_idx].numpy()
        C = tensors[f'correct_hidden_states_{window}'][layer_idx].numpy()
        M = tensors[f'misleading_hidden_states_{window}'][layer_idx].numpy()
        I = tensors[f'irrelevant_hidden_states_{window}'][layer_idx].numpy()
        
        # No text vs all text variants
        X_notext.append(N)
        X_text.extend([C, M, I])
        
        # All from same question get same group ID
        groups.extend([q_id] * 4)
    
    X = np.vstack([X_notext, X_text])
    y = np.array([0] * len(X_notext) + [1] * len(X_text))
    groups = np.array(groups)
    
    return X, y, groups



def extract_features_malicious(data, layer_idx, window):
    """
    Extract malicious detection features: M vs {N, C, I}
    
    Tests: Does misleading text create a unique representation 
    distinct from all other conditions?
    
    Returns:
        X: [1416, 4096] - features (354 misleading + 1062 other)
        y: [1416] - labels (1=misleading, 0=other)
        groups: [1416] - question IDs
    """
    X_misleading = []
    X_other = []
    groups = []
    
    for q_id, tensors in enumerate(data):
        N = tensors[f'notext_hidden_states_{window}'][layer_idx].numpy()
        C = tensors[f'correct_hidden_states_{window}'][layer_idx].numpy()
        M = tensors[f'misleading_hidden_states_{window}'][layer_idx].numpy()
        I = tensors[f'irrelevant_hidden_states_{window}'][layer_idx].numpy()
        
        # Misleading vs all others
        X_misleading.append(M)
        X_other.extend([N, C, I])
        
        # All from same question get same group ID
        groups.extend([q_id] * 4)
    
    X = np.vstack([X_misleading, X_other])
    y = np.array([1] * len(X_misleading) + [0] * len(X_other))
    groups = np.array(groups)
    
    return X, y, groups

def extract_features_relevance(data, layer_idx, window):
    """
    Extract relevance features: {C, M} vs I
    
    Tests: Does the model distinguish task-relevant from irrelevant text?
    
    Returns:
        X: [1062, 4096] - features (708 relevant + 354 irrelevant)
        y: [1062] - labels (1=relevant, 0=irrelevant)
        groups: [1062] - question IDs
    """
    X_relevant = []
    X_irrelevant = []
    groups = []
    
    for q_id, tensors in enumerate(data):
        C = tensors[f'correct_hidden_states_{window}'][layer_idx].numpy()
        M = tensors[f'misleading_hidden_states_{window}'][layer_idx].numpy()
        I = tensors[f'irrelevant_hidden_states_{window}'][layer_idx].numpy()
        
        # Relevant (C, M) vs Irrelevant (I)
        X_relevant.extend([C, M])
        X_irrelevant.append(I)
        
        # All from same question get same group ID
        groups.extend([q_id] * 3)
    
    X = np.vstack([X_relevant, X_irrelevant])
    y = np.array([1] * len(X_relevant) + [0] * len(X_irrelevant))
    groups = np.array(groups)
    
    return X, y, groups


def extract_features_correctness(data, layer_idx, window):
    """
    Extract correctness features: C vs M
    
    Tests: Does the model distinguish correct from misleading text?
    
    Returns:
        X: [708, 4096] - features
        y: [708] - labels (1=correct, 0=misleading)
        groups: [708] - question IDs
    """
    X_correct = []
    X_misleading = []
    groups = []
    
    for q_id, tensors in enumerate(data):
        C = tensors[f'correct_hidden_states_{window}'][layer_idx].numpy()
        M = tensors[f'misleading_hidden_states_{window}'][layer_idx].numpy()
        
        X_correct.append(C)
        X_misleading.append(M)
        groups.extend([q_id, q_id])
    
    X = np.vstack([X_correct, X_misleading])
    y = np.array([1] * len(X_correct) + [0] * len(X_misleading))
    groups = np.array(groups)
    
    return X, y, groups


# ============================================================================
# Probe Training
# ============================================================================

def train_probe(X, y, groups, cv_folds=5, random_seed=42):
    """
    Train probe with group-aware cross-validation.
    
    Returns:
        dict with auc_mean, auc_std, auc_scores
    """
    # Create classifier with balanced class weights
    clf = LogisticRegression(
        max_iter=1000, 
        random_state=random_seed,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Group-aware cross-validation
    cv = GroupKFold(n_splits=cv_folds)
    
    # Train and evaluate on each fold
    auc_scores = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred = clf.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)
    
    return {
        'auc_mean': float(np.mean(auc_scores)),
        'auc_std': float(np.std(auc_scores)),
        'auc_scores': [float(s) for s in auc_scores]
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_probe_analysis(data, probe_type, window, config):
    """
    Run one probe configuration across all layers.
    
    Args:
        data: Loaded activation data
        probe_type: 'detection', 'relevance', or 'correctness'
        window: Token window name
        config: Configuration object
    
    Returns:
        List of results per layer
    """
    print(f"\n{'='*70}")
    print(f"Running: {probe_type} - {window}")
    print('='*70)
    
    results = []
    
    for layer in tqdm(range(config.NUM_LAYERS), desc=f"Layers"):
        # Extract features based on probe type
        if probe_type == 'detection':
            X, y, groups = extract_features_detection(data, layer, window)
        elif probe_type == 'relevance':
            X, y, groups = extract_features_relevance(data, layer, window)
        elif probe_type == 'correctness':
            X, y, groups = extract_features_correctness(data, layer, window)
        elif probe_type == 'malicious':
            X, y, groups = extract_features_malicious(data, layer, window)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")
        
        # Train probe
        result = train_probe(X, y, groups, config.CV_FOLDS, config.RANDOM_SEED)
        result['layer'] = layer
        results.append(result)
    
    # Print summary for this configuration
    best_layer = max(results, key=lambda x: x['auc_mean'])
    print(f"✓ Best: Layer {best_layer['layer']}, AUC = {best_layer['auc_mean']:.3f} ± {best_layer['auc_std']:.3f}")
    
    return results


def run_all_probes(data, config):
    """
    Run all 3 essential probe configurations × 4 windows = 12 total.
    
    Returns:
        Dict with all results
    """
    all_results = {
        'detection': {},
        'relevance': {},
        'correctness': {},
        'malicious': {}
    }
    
    print("\n" + "="*70)
    print("RUNNING ESSENTIAL PROBES")
    print("="*70)
    print(f"Total: 3 probe types × 4 windows × {config.NUM_LAYERS} layers = {3*4*config.NUM_LAYERS} configurations")
    print("\nProbe Hierarchy:")
    print("  1. Detection:   N vs {C, M, I} - Text presence")
    print("  2. Relevance:   {C, M} vs I    - Task relevance")
    print("  3. Correctness: C vs M         - Semantic correctness")
    print("  4. Malicious:   M vs {N, C, I} - Misleading detection")
    
    # Run each probe type on each window
    # Run each probe type on appropriate windows
    for probe_type in ['detection', 'relevance', 'correctness', 'malicious']:
        if probe_type == 'detection':
            # Detection runs on all windows
            windows_to_run = config.WINDOWS
        else:
            # Other probes only on single_token
            windows_to_run = ['single_token']
        
        for window in windows_to_run:
            results = run_probe_analysis(data, probe_type, window, config)
            all_results[probe_type][window] = results
    
    return all_results


# ============================================================================
# Save Results
# ============================================================================

def save_results(results, output_dir):
    """Save results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save each probe type × window combination
    for probe_type in results:
        for window in results[probe_type]:
            filename = f"{probe_type}_{window}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(results[probe_type][window], f, indent=2)
            
            print(f"✓ Saved: {filename}")
    
    # Save combined summary
    summary_path = output_dir / "all_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: all_results.json")


# ============================================================================
# Summary Statistics
# ============================================================================

def print_summary(results):
    """Print summary of all results."""
    print("\n" + "="*70)
    print("SUMMARY - ESSENTIAL PROBES")
    print("="*70)
    
    for probe_type in ['detection', 'relevance', 'correctness']:
        print(f"\n{probe_type.upper()}:")
        print("-" * 70)
        
        for window in results[probe_type]:
            layer_results = results[probe_type][window]
            best = max(layer_results, key=lambda x: x['auc_mean'])
            
            print(f"  {window:20s}: Layer {best['layer']:2d}, AUC = {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")
    
    # Key comparisons
    print("\n" + "="*70)
    print("PROCESSING HIERARCHY (at single_token)")
    print("="*70)
    
    # At last token - the most interpretable window
    last_token = 'single_token'
    det = max(results['detection'][last_token], key=lambda x: x['auc_mean'])
    rel = max(results['relevance'][last_token], key=lambda x: x['auc_mean'])
    cor = max(results['correctness'][last_token], key=lambda x: x['auc_mean'])
    
    print(f"\n  Detection (N vs {{C,M,I}}):    {det['auc_mean']:.3f} at Layer {det['layer']}")
    print(f"  Relevance ({{C,M}} vs I):      {rel['auc_mean']:.3f} at Layer {rel['layer']}")
    print(f"  Correctness (C vs M):        {cor['auc_mean']:.3f} at Layer {cor['layer']}")
    
    # Interpretation guide
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nExpected patterns:")
    print("  • Detection should peak early (low/mid layers)")
    print("  • Relevance should peak in middle layers")
    print("  • Correctness should peak late (high layers)")
    print("\nIf correctness AUC is low:")
    print("  → Model may not process semantic meaning of text")
    print("If relevance > correctness:")
    print("  → Model attends to text but doesn't verify accuracy")
    print("If detection ≈ relevance ≈ correctness:")
    print("  → All processing may collapse into simple text detection")
    
    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution."""
    config = Config()
    
    print("="*70)
    print("LINEAR PROBE ANALYSIS - ESSENTIAL SUITE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Activations: {config.ACTIVATIONS_DIR}")
    print(f"  Output:      {config.OUTPUT_DIR}")
    print(f"  Windows:     {', '.join(config.WINDOWS)}")
    print(f"  Layers:      {config.NUM_LAYERS}")
    print(f"  CV Folds:    {config.CV_FOLDS}")
    print(f"  Random Seed: {config.RANDOM_SEED}")
    
    # Load data
    data = load_all_activations(config.ACTIVATIONS_DIR)
    
    # Run all probes
    results = run_all_probes(data, config)
    
    # Save results
    save_results(results, config.OUTPUT_DIR)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {config.OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  • Plot AUC curves across layers for each probe")
    print("  • Compare layer progression: detection → relevance → correctness")
    print("  • Identify where semantic processing emerges vs collapses")


if __name__ == "__main__":
    main()