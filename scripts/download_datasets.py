#!/usr/bin/env python3
"""
Download public datasets for realistic workload patterns.
Supports: Alibaba Cluster Trace v2018, Azure Public Dataset v2
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Dataset URLs and metadata
DATASETS = {
    "alibaba_2018_sample": {
        "name": "Alibaba Cluster Trace 2018 (Sample)",
        "description": "Sample container usage data from Alibaba production cluster",
        "url": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2018/sample_container_usage.csv",
        "size": "~5MB (sample)",
        "target": "data/external/alibaba_2018_sample.csv",
    },
    "azure_v2_sample": {
        "name": "Azure Public Dataset v2 (Sample)",
        "description": "Sample VM CPU utilization from Azure",
        "url": "https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.csv",
        "size": "~100MB",
        "target": "data/external/azure_functions_2019.csv",
    },
}

# Note: Full datasets require registration/survey
FULL_DATASETS_INFO = """
==============================================================================
FULL DATASETS (Require registration):
==============================================================================

1. Alibaba Cluster Trace v2018 (FULL - 280GB uncompressed)
   Registration: https://github.com/alibaba/clusterdata
   After survey approval, download links:
   - container_usage.tar.gz (28GB) - CPU/Memory every 5 min
   - container_meta.tar.gz (2.4MB) - Container metadata

2. Azure Public Dataset v2 (FULL - 200GB+)
   Direct download (no registration):
   - https://github.com/Azure/AzurePublicDataset
   - VM CPU utilization readings (2.6M VMs, 1.9B readings)

3. Google Cluster Workload Traces 2019 (2.4TB)
   Access via BigQuery only:
   - https://github.com/google/cluster-data
   - Requires Google Cloud account + BigQuery credits

==============================================================================
"""


def download_with_progress(url: str, target: Path) -> None:
    """Download file with progress reporting."""

    def report_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(block_num * block_size / total_size * 100, 100)
            sys.stdout.write(
                f"\r  Progress: {percent:.1f}% ({block_num * block_size / 1024 / 1024:.1f} MB)"
            )
            sys.stdout.flush()

    print(f"Downloading: {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target, reporthook=report_hook)
    print()  # New line after progress


def download_dataset(dataset_key: str) -> Path:
    """Download a specific dataset."""

    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")

    dataset = DATASETS[dataset_key]
    target = Path(dataset["target"])

    if target.exists():
        print(f"✓ Dataset already exists: {target}")
        file_size = target.stat().st_size / 1024 / 1024
        print(f"  Size: {file_size:.1f} MB")
        return target

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset['name']}")
    print(f"Description: {dataset['description']}")
    print(f"Expected size: {dataset['size']}")
    print(f"{'='*70}\n")

    try:
        download_with_progress(dataset["url"], target)
        print(f"✓ Downloaded successfully: {target}")
        return target
    except Exception as e:
        print(f"✗ Download failed: {e}")
        if target.exists():
            target.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download public datasets for K8s autoscaling research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=FULL_DATASETS_INFO,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all available samples)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about full datasets",
    )

    args = parser.parse_args()

    if args.info:
        print(FULL_DATASETS_INFO)
        return

    print("\n" + "=" * 70)
    print("PUBLIC DATASET DOWNLOADER")
    print("=" * 70)

    datasets_to_download = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    downloaded = []
    failed = []

    for dataset_key in datasets_to_download:
        try:
            path = download_dataset(dataset_key)
            downloaded.append((dataset_key, path))
        except Exception as e:
            failed.append((dataset_key, str(e)))

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    if downloaded:
        print(f"\n✓ Successfully downloaded ({len(downloaded)}):")
        for key, path in downloaded:
            size = path.stat().st_size / 1024 / 1024
            print(f"  - {key}: {path} ({size:.1f} MB)")

    if failed:
        print(f"\n✗ Failed downloads ({len(failed)}):")
        for key, error in failed:
            print(f"  - {key}: {error}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run converter: python scripts/convert_datasets.py")
    print("2. Check converted data: ls -lh data/processed/")
    print("3. Train models: python -m k8s_ml_predictive_autoscaling.predictor.train")
    print("\nFor FULL datasets, see: python scripts/download_datasets.py --info")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
