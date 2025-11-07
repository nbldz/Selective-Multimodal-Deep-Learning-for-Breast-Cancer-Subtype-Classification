"""
Data Preparation Utilities
Helper functions for preparing TCGA-BRCA data for training
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse


class DataValidator:
    """Validates data integrity and structure"""
    
    def __init__(self, brca_wsi_dir: str, rna_csv_path: str):
        self.brca_wsi_dir = Path(brca_wsi_dir)
        self.rna_csv_path = Path(rna_csv_path)
        
    def validate_directory_structure(self) -> Dict:
        """Validate WSI directory structure"""
        print("Validating WSI directory structure...")
        
        if not self.brca_wsi_dir.exists():
            raise ValueError(f"BRCA directory not found: {self.brca_wsi_dir}")
        
        # Get all patient directories
        patient_dirs = [d for d in self.brca_wsi_dir.iterdir() if d.is_dir()]
        
        results = {
            'total_patients': len(patient_dirs),
            'patients_with_patches': 0,
            'patients_without_patches': 0,
            'total_patches': 0,
            'patients': {}
        }
        
        for patient_dir in tqdm(patient_dirs, desc="Checking patients"):
            patient_id = patient_dir.name
            patches = list(patient_dir.glob("*.jpg"))
            
            results['patients'][patient_id] = {
                'num_patches': len(patches),
                'has_patches': len(patches) > 0
            }
            
            if len(patches) > 0:
                results['patients_with_patches'] += 1
                results['total_patches'] += len(patches)
            else:
                results['patients_without_patches'] += 1
        
        print(f"\nDirectory Validation Results:")
        print(f"  Total patients: {results['total_patients']}")
        print(f"  Patients with patches: {results['patients_with_patches']}")
        print(f"  Patients without patches: {results['patients_without_patches']}")
        print(f"  Total patches: {results['total_patches']}")
        print(f"  Avg patches per patient: {results['total_patches'] / max(results['patients_with_patches'], 1):.1f}")
        
        return results
    
    def validate_rna_csv(self) -> Dict:
        """Validate RNA-seq CSV file"""
        print("\nValidating RNA-seq CSV...")
        
        if not self.rna_csv_path.exists():
            raise ValueError(f"RNA CSV not found: {self.rna_csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.rna_csv_path)
        
        results = {
            'total_samples': len(df),
            'num_features': len(df.columns) - 2,  # Excluding patient_id and PAM50
            'missing_values': df.isnull().sum().sum(),
            'class_distribution': {}
        }
        
        # Check required columns
        if 'patient_id' not in df.columns:
            raise ValueError("CSV must contain 'patient_id' column")
        
        if 'PAM50' not in df.columns:
            raise ValueError("CSV must contain 'PAM50' column")
        
        # Check class distribution
        if 'PAM50' in df.columns:
            results['class_distribution'] = df['PAM50'].value_counts().to_dict()
        
        print(f"\nRNA CSV Validation Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Number of genes: {results['num_features']}")
        print(f"  Missing values: {results['missing_values']}")
        print(f"\n  Class distribution:")
        for class_name, count in results['class_distribution'].items():
            print(f"    {class_name}: {count} ({count/results['total_samples']*100:.1f}%)")
        
        return results
    
    def check_patient_alignment(self) -> Dict:
        """Check alignment between RNA CSV and WSI directories"""
        print("\nChecking patient alignment...")
        
        # Load RNA CSV
        rna_df = pd.read_csv(self.rna_csv_path)
        rna_patients = set(rna_df['patient_id'].values)
        
        # Get WSI patients
        wsi_patients = set([d.name for d in self.brca_wsi_dir.iterdir() if d.is_dir()])
        
        # Find matches
        matched_patients = rna_patients.intersection(wsi_patients)
        rna_only = rna_patients - wsi_patients
        wsi_only = wsi_patients - rna_patients
        
        results = {
            'rna_patients': len(rna_patients),
            'wsi_patients': len(wsi_patients),
            'matched_patients': len(matched_patients),
            'rna_only_patients': len(rna_only),
            'wsi_only_patients': len(wsi_only),
            'alignment_rate': len(matched_patients) / len(rna_patients) * 100
        }
        
        print(f"\nPatient Alignment Results:")
        print(f"  RNA patients: {results['rna_patients']}")
        print(f"  WSI patients: {results['wsi_patients']}")
        print(f"  Matched patients: {results['matched_patients']}")
        print(f"  RNA-only patients: {results['rna_only_patients']}")
        print(f"  WSI-only patients: {results['wsi_only_patients']}")
        print(f"  Alignment rate: {results['alignment_rate']:.1f}%")
        
        if results['alignment_rate'] < 80:
            print("\n  ⚠️  Warning: Low alignment rate. Check patient IDs.")
        
        return results, matched_patients
    
    def validate_patch_quality(self, sample_size: int = 100) -> Dict:
        """Check patch image quality"""
        print(f"\nValidating patch quality (sampling {sample_size} patches)...")
        
        # Get random patches
        all_patches = []
        for patient_dir in self.brca_wsi_dir.iterdir():
            if patient_dir.is_dir():
                patches = list(patient_dir.glob("*.jpg"))
                all_patches.extend(patches)
        
        if not all_patches:
            raise ValueError("No patches found")
        
        # Sample patches
        import random
        sample_patches = random.sample(all_patches, min(sample_size, len(all_patches)))
        
        results = {
            'valid_patches': 0,
            'corrupted_patches': 0,
            'sizes': [],
            'modes': []
        }
        
        for patch_path in tqdm(sample_patches, desc="Checking patches"):
            try:
                img = Image.open(patch_path)
                results['valid_patches'] += 1
                results['sizes'].append(img.size)
                results['modes'].append(img.mode)
            except Exception as e:
                results['corrupted_patches'] += 1
                print(f"  Corrupted: {patch_path}")
        
        # Summarize sizes
        unique_sizes = list(set(results['sizes']))
        size_counts = {size: results['sizes'].count(size) for size in unique_sizes}
        
        print(f"\nPatch Quality Results:")
        print(f"  Valid patches: {results['valid_patches']}/{sample_size}")
        print(f"  Corrupted patches: {results['corrupted_patches']}")
        print(f"  Image sizes: {size_counts}")
        print(f"  Image modes: {set(results['modes'])}")
        
        return results


class DatasetCreator:
    """Creates properly formatted datasets"""
    
    @staticmethod
    def create_filtered_csv(
        input_csv: str,
        output_csv: str,
        matched_patients: set,
        min_patches: int = 5
    ):
        """Create filtered CSV with only patients that have WSI data"""
        print(f"\nCreating filtered CSV...")
        
        df = pd.read_csv(input_csv)
        
        # Filter to matched patients
        df_filtered = df[df['patient_id'].isin(matched_patients)]
        
        # Save
        df_filtered.to_csv(output_csv, index=False)
        
        print(f"  Original samples: {len(df)}")
        print(f"  Filtered samples: {len(df_filtered)}")
        print(f"  Saved to: {output_csv}")
        
        return df_filtered
    
    @staticmethod
    def create_data_splits(
        csv_path: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """Create train/val/test splits"""
        from sklearn.model_selection import train_test_split
        
        print(f"\nCreating data splits...")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(csv_path)
        
        # First split: train + temp
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            stratify=df['PAM50']
        )
        
        # Second split: val + test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed,
            stratify=temp_df['PAM50']
        )
        
        # Save splits
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")
        print(f"  Test samples: {len(test_df)}")
        print(f"  Saved to: {output_dir}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def create_sample_rna_json(csv_path: str, patient_id: str, output_path: str):
        """Create sample RNA JSON for inference"""
        df = pd.read_csv(csv_path)
        
        patient_row = df[df['patient_id'] == patient_id]
        
        if len(patient_row) == 0:
            raise ValueError(f"Patient {patient_id} not found")
        
        # Extract gene values
        gene_cols = [col for col in df.columns if col not in ['patient_id', 'PAM50']]
        gene_values = patient_row[gene_cols].iloc[0].to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(gene_values, f, indent=4)
        
        print(f"Sample RNA data saved to {output_path}")


def generate_data_report(brca_dir: str, rna_csv: str, output_path: str = "data_report.txt"):
    """Generate comprehensive data report"""
    validator = DataValidator(brca_dir, rna_csv)
    
    report = []
    report.append("="*80)
    report.append("TCGA-BRCA Data Validation Report")
    report.append("="*80)
    report.append("")
    
    # Validate directory
    dir_results = validator.validate_directory_structure()
    
    # Validate RNA CSV
    rna_results = validator.validate_rna_csv()
    
    # Check alignment
    alignment_results, matched_patients = validator.check_patient_alignment()
    
    # Validate patches
    try:
        patch_results = validator.validate_patch_quality(sample_size=100)
    except Exception as e:
        patch_results = None
        report.append(f"Patch validation failed: {e}")
    
    # Summary
    report.append("\n" + "="*80)
    report.append("SUMMARY")
    report.append("="*80)
    report.append(f"✓ Dataset ready: {alignment_results['alignment_rate'] > 80}")
    report.append(f"✓ Total usable patients: {alignment_results['matched_patients']}")
    report.append(f"✓ Average patches per patient: {dir_results['total_patches'] / max(dir_results['patients_with_patches'], 1):.1f}")
    
    if rna_results['missing_values'] > 0:
        report.append(f"⚠️  Warning: {rna_results['missing_values']} missing values in RNA data")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nFull report saved to: {output_path}")
    
    return matched_patients


def main():
    parser = argparse.ArgumentParser(description='Data preparation utilities for BRCA classification')
    parser.add_argument('--brca_dir', type=str, required=True, help='Path to BRCA WSI directory')
    parser.add_argument('--rna_csv', type=str, required=True, help='Path to RNA-seq CSV')
    parser.add_argument('--output_dir', type=str, default='./prepared_data', help='Output directory')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation')
    parser.add_argument('--create_splits', action='store_true', help='Create train/val/test splits')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate report and get matched patients
    print("Running data validation...")
    matched_patients = generate_data_report(
        args.brca_dir,
        args.rna_csv,
        os.path.join(args.output_dir, 'data_report.txt')
    )
    
    if not args.validate_only:
        # Create filtered CSV
        filtered_csv = os.path.join(args.output_dir, 'filtered_data.csv')
        DatasetCreator.create_filtered_csv(
            args.rna_csv,
            filtered_csv,
            matched_patients
        )
        
        if args.create_splits:
            # Create splits
            DatasetCreator.create_data_splits(
                filtered_csv,
                args.output_dir
            )
    
    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review the data report: {args.output_dir}/data_report.txt")
    print(f"2. Use filtered CSV for training: {args.output_dir}/filtered_data.csv")
    if args.create_splits:
        print(f"3. Train/val/test splits ready in: {args.output_dir}/")
    print(f"4. Update paths in main.py and start training!")


if __name__ == "__main__":
    main()