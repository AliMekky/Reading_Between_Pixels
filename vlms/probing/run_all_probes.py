#!/usr/bin/env python3
"""
Linear Probe Analysis for VLM Text Processing

Simple, clean implementation of 12 probes:
- 4 Semantic (Direction Vectors) (I - C) vs (I - M) 
- 4 Semantic (Raw Activations) (C vs M)
- 4 Detection (N vs I)

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
    ACTIVATIONS_DIR = "./activations_fixed"
    OUTPUT_DIR = "./probe_analysis"
    
    WINDOWS = ['single_token', 'decision_last10', 'after_vision', 'vision_tokens']
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

def extract_features_semantic_directions(data, layer_idx, window):
    """
    Extract semantic direction features: (C-I) vs (M-I)
    
    Returns:
        X: [708, 4096] - features
        y: [708] - labels (1=correct, 0=misleading)
        groups: [708] - question IDs
    """
    X_correct = []
    X_misleading = []
    groups = []
    
    for q_id, tensors in enumerate(data):
        # Get representations at this layer for this window
        I = tensors[f'irrelevant_hidden_states_{window}'][layer_idx].numpy()
        C = tensors[f'correct_hidden_states_{window}'][layer_idx].numpy()
        M = tensors[f'misleading_hidden_states_{window}'][layer_idx].numpy()
        
        # Compute semantic directions
        X_correct.append(C - I)
        X_misleading.append(M - I)
        
        # Both examples from same question get same group ID
        groups.extend([q_id, q_id])
    
    # Stack into arrays
    X = np.vstack([X_correct, X_misleading])
    y = np.array([1] * len(X_correct) + [0] * len(X_misleading))
    groups = np.array(groups)
    
    return X, y, groups


def extract_features_semantic_raw(data, layer_idx, window):
    """
    Extract raw semantic features: C vs M
    
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


def extract_features_detection(data, layer_idx, window):
    """
    Extract detection features: N vs I
    
    Returns:
        X: [708, 4096] - features
        y: [708] - labels (0=no text, 1=has text)
        groups: [708] - question IDs
    """
    X_notext = []
    X_text = []
    groups = []
    
    for q_id, tensors in enumerate(data):
        N = tensors[f'notext_hidden_states_{window}'][layer_idx].numpy()
        I = tensors[f'irrelevant_hidden_states_{window}'][layer_idx].numpy()
        
        X_notext.append(N)
        X_text.append(I)
        groups.extend([q_id, q_id])
    
    X = np.vstack([X_notext, X_text])
    y = np.array([0] * len(X_notext) + [1] * len(X_text))
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
    # Create classifier
    clf = LogisticRegression(max_iter=1000, random_state=random_seed)
    
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
        probe_type: 'semantic_directions', 'semantic_raw', or 'detection'
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
        if probe_type == 'semantic_directions':
            X, y, groups = extract_features_semantic_directions(data, layer, window)
        elif probe_type == 'semantic_raw':
            X, y, groups = extract_features_semantic_raw(data, layer, window)
        elif probe_type == 'detection':
            X, y, groups = extract_features_detection(data, layer, window)
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
    Run all 12 probe configurations.
    
    Returns:
        Dict with all results
    """
    all_results = {
        'semantic_directions': {},
        'semantic_raw': {},
        'detection': {}
    }
    
    print("\n" + "="*70)
    print("RUNNING ALL PROBES")
    print("="*70)
    print(f"Total: 3 probe types × 4 windows × {config.NUM_LAYERS} layers = {3*4*config.NUM_LAYERS} configurations")
    
    # Run each probe type on each window
    for probe_type in ['semantic_directions', 'semantic_raw', 'detection']:
        for window in config.WINDOWS:
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
    print("SUMMARY")
    print("="*70)
    
    for probe_type in ['semantic_directions', 'semantic_raw', 'detection']:
        print(f"\n{probe_type.upper().replace('_', ' ')}:")
        print("-" * 70)
        
        for window in results[probe_type]:
            layer_results = results[probe_type][window]
            best = max(layer_results, key=lambda x: x['auc_mean'])
            
            print(f"  {window:20s}: Layer {best['layer']:2d}, AUC = {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")
    
    # Key comparisons
    print("\n" + "="*70)
    print("KEY COMPARISONS")
    print("="*70)
    
    # At last token
    last_token = 'single_token'
    sem_dir = max(results['semantic_directions'][last_token], key=lambda x: x['auc_mean'])
    sem_raw = max(results['semantic_raw'][last_token], key=lambda x: x['auc_mean'])
    det = max(results['detection'][last_token], key=lambda x: x['auc_mean'])
    
    print(f"\nAt Last Token:")
    print(f"  Semantic (Directions): {sem_dir['auc_mean']:.3f} at Layer {sem_dir['layer']}")
    print(f"  Semantic (Raw):        {sem_raw['auc_mean']:.3f} at Layer {sem_raw['layer']}")
    print(f"  Detection:             {det['auc_mean']:.3f} at Layer {det['layer']}")
    
    # At after vision
    after_vis = 'after_vision'
    sem_dir_av = max(results['semantic_directions'][after_vis], key=lambda x: x['auc_mean'])
    det_av = max(results['detection'][after_vis], key=lambda x: x['auc_mean'])
    
    print(f"\nAt After Vision:")
    print(f"  Semantic (Directions): {sem_dir_av['auc_mean']:.3f} at Layer {sem_dir_av['layer']}")
    print(f"  Detection:             {det_av['auc_mean']:.3f} at Layer {det_av['layer']}")
    
    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution."""
    config = Config()
    
    print("="*70)
    print("LINEAR PROBE ANALYSIS - COMPREHENSIVE SUITE")
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


if __name__ == "__main__":
    main()