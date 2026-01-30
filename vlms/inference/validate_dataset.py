"""
HuggingFace Dataset Image Validator
Checks all image variants for corruption, missing files, and other issues.
"""

import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict
from PIL import Image, ImageFile
import json
from tqdm import tqdm

# Enable truncated image loading for diagnosis
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        print(f"✓ Loading from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"⬇️  Downloading '{dataset_id}'...")
    ds = load_dataset(dataset_id, split=split)
    
    try:
        ds.save_to_disk(str(cache_dir))
        print(f"✓ Saved to cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️  Cache save failed: {e}")
    
    return ds


def check_image(img_obj, variant_name, sample_idx, question_id):
    """
    Check if an image is valid and can be loaded.
    
    Returns:
        dict with status and error info
    """
    result = {
        "sample_idx": sample_idx,
        "question_id": question_id,
        "variant": variant_name,
        "status": "ok",
        "error": None,
        "image_type": type(img_obj).__name__
    }
    
    try:
        # Case 1: Already a PIL Image
        if isinstance(img_obj, Image.Image):
            img_obj.load()  # Force load
            result["width"] = img_obj.width
            result["height"] = img_obj.height
            result["mode"] = img_obj.mode
            return result
        
        # Case 2: Dict with 'path'
        elif isinstance(img_obj, dict) and 'path' in img_obj:
            img_path = img_obj['path']
            result["path"] = img_path
            
            if not Path(img_path).exists():
                result["status"] = "missing"
                result["error"] = f"File not found: {img_path}"
                return result
            
            img = Image.open(img_path)
            img.load()
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode
            return result
        
        # Case 3: Dict with 'bytes'
        elif isinstance(img_obj, dict) and 'bytes' in img_obj:
            import io
            img = Image.open(io.BytesIO(img_obj['bytes']))
            img.load()
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode
            return result
        
        # Case 4: None or missing
        elif img_obj is None:
            result["status"] = "missing"
            result["error"] = "Image object is None"
            return result
        
        else:
            result["status"] = "unknown_type"
            result["error"] = f"Unknown image type: {type(img_obj)}"
            return result
            
    except OSError as e:
        result["status"] = "corrupted"
        result["error"] = f"OSError: {str(e)}"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        return result


def validate_dataset(dataset_id, cache_dir, split="test", output_file=None):
    """
    Validate all images in the dataset across all variants.
    
    Returns:
        dict with validation statistics and problematic samples
    """
    print(f"\n{'='*60}")
    print(f"Validating Dataset: {dataset_id}")
    print(f"{'='*60}\n")
    
    # Load dataset
    ds = get_or_download_hf_dataset(dataset_id, cache_dir, split)
    
    if isinstance(ds, DatasetDict):
        split_name = split if split in ds else list(ds.keys())[0]
        dataset = ds[split_name]
    else:
        dataset = ds
    
    print(f"Dataset size: {len(dataset)} samples\n")
    
    # Variants to check
    variants = ['notext', 'correct', 'irrelevant', 'misleading']
    
    # Statistics
    stats = {
        "total_samples": len(dataset),
        "variants_checked": variants,
        "results_by_variant": {v: {"ok": 0, "corrupted": 0, "missing": 0, "error": 0} for v in variants},
        "problematic_samples": []
    }
    
    # Check each sample
    print("Checking images...")
    for idx in tqdm(range(len(dataset)), desc="Validating"):
        try:
            sample = dataset[idx]
            question_id = sample.get("question_id", f"unknown_{idx}")
            
            sample_issues = []
            
            for variant in variants:
                img_obj = sample.get(variant)
                result = check_image(img_obj, variant, idx, question_id)
                
                # Update stats
                stats["results_by_variant"][variant][result["status"]] += 1
                
                # Track problematic samples
                if result["status"] != "ok":
                    sample_issues.append(result)
            
            # If any issues, add to problematic list
            if sample_issues:
                stats["problematic_samples"].append({
                    "sample_idx": idx,
                    "question_id": question_id,
                    "issues": sample_issues
                })
                
        except Exception as e:
            print(f"\n⚠️  Error accessing sample {idx}: {e}")
            stats["problematic_samples"].append({
                "sample_idx": idx,
                "question_id": "unknown",
                "issues": [{"error": str(e), "status": "access_error"}]
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"Total samples: {stats['total_samples']}\n")
    
    for variant in variants:
        v_stats = stats["results_by_variant"][variant]
        print(f"{variant.upper()}:")
        print(f"  ✓ OK:        {v_stats['ok']}")
        print(f"  ✗ Corrupted: {v_stats['corrupted']}")
        print(f"  ✗ Missing:   {v_stats['missing']}")
        print(f"  ✗ Error:     {v_stats['error']}")
        print()
    
    # Problematic samples summary
    num_problematic = len(stats["problematic_samples"])
    print(f"Problematic samples: {num_problematic}/{stats['total_samples']}")
    
    if num_problematic > 0:
        print(f"\n{'='*60}")
        print("PROBLEMATIC SAMPLES (first 10):")
        print(f"{'='*60}\n")
        
        for sample_info in stats["problematic_samples"][:10]:
            print(f"Sample {sample_info['sample_idx']} (ID: {sample_info['question_id']}):")
            for issue in sample_info['issues']:
                print(f"  - {issue['variant']}: {issue['status']} - {issue.get('error', 'N/A')}")
            print()
    
    # Save detailed report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Detailed report saved to: {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate HuggingFace dataset images")
    
    parser.add_argument(
        '--hf_dataset', 
        type=str, 
        required=True,
        help='HuggingFace dataset ID (e.g., AHAAM/CIM)'
    )
    parser.add_argument(
        '--hf_cache_dir', 
        type=str, 
        default='./hf_dataset_local_cache',
        help='Local cache directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to validate (default: test)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_report.json',
        help='Output JSON file for detailed report'
    )
    
    args = parser.parse_args()
    
    stats = validate_dataset(
        dataset_id=args.hf_dataset,
        cache_dir=args.hf_cache_dir,
        split=args.split,
        output_file=args.output
    )
    
    # Exit code based on results
    num_problematic = len(stats["problematic_samples"])
    if num_problematic == 0:
        print("\n✓ All images validated successfully!")
        return 0
    else:
        print(f"\n⚠️  Found {num_problematic} samples with issues")
        return 1


if __name__ == "__main__":
    exit(main())