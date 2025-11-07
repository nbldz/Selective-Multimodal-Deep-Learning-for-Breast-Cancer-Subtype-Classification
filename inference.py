"""
Inference script for breast cancer subtype classification
Use trained models to make predictions on new samples
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

# Import from main script
from main import (
    Config, RNAOnlyClassifier, WSIOnlyClassifier, 
    MultimodalClassifier, DataPreprocessor, RoutingSystem
)


class InferenceEngine:
    """Handles inference on new samples"""
    
    def __init__(self, config_path=None):
        self.config = Config()
        
        # Load models
        self.device = self.config.DEVICE
        self.models = {}
        self.preprocessor = None
        
    def load_models(self, model_dir):
        """Load all trained models"""
        print("Loading models...")
        
        # Load RNA model
        rna_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(rna_model_path):
            self.models['rna'] = RNAOnlyClassifier(
                input_dim=self.config.NUM_TOP_GENES,
                hidden_dim=self.config.RNA_HIDDEN_DIM,
                num_classes=self.config.NUM_CLASSES
            ).to(self.device)
            self.models['rna'].load_state_dict(torch.load(rna_model_path))
            self.models['rna'].eval()
            print("✓ RNA model loaded")
        
        # Load multimodal models
        for fusion_type in ['concatenation', 'gated', 'cross_attention']:
            model_path = os.path.join(model_dir, f"multimodal_{fusion_type}.pth")
            if os.path.exists(model_path):
                self.models[f'multimodal_{fusion_type}'] = MultimodalClassifier(
                    rna_input_dim=self.config.NUM_TOP_GENES,
                    rna_hidden_dim=self.config.RNA_HIDDEN_DIM,
                    wsi_feature_dim=self.config.WSI_FEATURE_DIM,
                    fusion_hidden_dim=self.config.FUSION_HIDDEN_DIM,
                    num_classes=self.config.NUM_CLASSES,
                    fusion_type=fusion_type
                ).to(self.device)
                self.models[f'multimodal_{fusion_type}'].load_state_dict(
                    torch.load(model_path)
                )
                self.models[f'multimodal_{fusion_type}'].eval()
                print(f"✓ Multimodal ({fusion_type}) model loaded")
        
        # Initialize routing system
        if 'rna' in self.models and 'multimodal_cross_attention' in self.models:
            self.routing = RoutingSystem(
                self.models['rna'],
                self.models['multimodal_cross_attention'],
                self.device
            )
            print("✓ Routing system initialized")
    
    def load_preprocessor(self, preprocessor_path):
        """Load fitted preprocessor"""
        import pickle
        
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print("✓ Preprocessor loaded")
        else:
            print("Warning: Preprocessor not found, initializing new one")
            self.preprocessor = DataPreprocessor(self.config)
    
    def preprocess_rna(self, rna_data):
        """Preprocess RNA-seq data for a single sample"""
        if isinstance(rna_data, pd.Series):
            rna_data = rna_data.values
        
        # Apply same preprocessing as training
        rna_data = np.log1p(rna_data)
        rna_data = self.preprocessor.scaler.transform(rna_data.reshape(1, -1))
        
        return torch.FloatTensor(rna_data).to(self.device)
    
    def load_wsi_patches(self, patient_id):
        """Load WSI patches for a patient"""
        patient_dir = Path(self.config.BRCA_WSI_DIR) / patient_id
        
        if not patient_dir.exists():
            raise ValueError(f"Patient directory not found: {patient_dir}")
        
        # Get all patches
        patch_paths = list(patient_dir.glob("*.jpg"))
        
        if not patch_paths:
            raise ValueError(f"No patches found for patient {patient_id}")
        
        # Limit patches
        if len(patch_paths) > self.config.MAX_PATCHES_PER_PATIENT:
            import random
            patch_paths = random.sample(patch_paths, self.config.MAX_PATCHES_PER_PATIENT)
        
        # Load and transform patches
        transform = transforms.Compose([
            transforms.Resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        patches = []
        for path in patch_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            patches.append(img_tensor)
        
        # Stack patches
        wsi_tensor = torch.stack(patches).unsqueeze(0).to(self.device)
        patch_count = torch.LongTensor([len(patches)]).to(self.device)
        
        return wsi_tensor, patch_count
    
    def predict_single_sample(
        self, 
        patient_id, 
        rna_data, 
        use_routing=True,
        return_all_predictions=False
    ):
        """
        Make prediction for a single sample
        
        Args:
            patient_id: Patient identifier
            rna_data: RNA-seq expression data
            use_routing: Whether to use routing mechanism
            return_all_predictions: Return predictions from all models
        
        Returns:
            results: Dictionary with predictions and confidences
        """
        # Preprocess RNA
        rna_tensor = self.preprocess_rna(rna_data)
        
        # Load WSI patches
        try:
            wsi_tensor, patch_count = self.load_wsi_patches(patient_id)
        except Exception as e:
            print(f"Warning: Could not load WSI data: {e}")
            wsi_tensor = None
            patch_count = None
        
        results = {
            'patient_id': patient_id,
            'predictions': {},
            'confidences': {},
            'probabilities': {}
        }
        
        with torch.no_grad():
            # RNA-only prediction
            if 'rna' in self.models:
                logits, _ = self.models['rna'](rna_tensor)
                probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = probs[0, pred_class].item()
                
                results['predictions']['rna'] = pred_class
                results['confidences']['rna'] = confidence
                results['probabilities']['rna'] = probs[0].cpu().numpy()
            
            # Multimodal predictions (if WSI available)
            if wsi_tensor is not None:
                for model_name in self.models:
                    if model_name.startswith('multimodal_'):
                        logits, _, _ = self.models[model_name](
                            rna_tensor, wsi_tensor, patch_count
                        )
                        probs = F.softmax(logits, dim=1)
                        pred_class = torch.argmax(logits, dim=1).item()
                        confidence = probs[0, pred_class].item()
                        
                        results['predictions'][model_name] = pred_class
                        results['confidences'][model_name] = confidence
                        results['probabilities'][model_name] = probs[0].cpu().numpy()
            
            # Routing prediction
            if use_routing and hasattr(self, 'routing') and wsi_tensor is not None:
                # Simple routing logic
                rna_confidence = results['confidences']['rna']
                
                if rna_confidence >= self.routing.optimal_threshold:
                    final_pred = results['predictions']['rna']
                    routing_used = False
                else:
                    final_pred = results['predictions']['multimodal_cross_attention']
                    routing_used = True
                
                results['predictions']['routing'] = final_pred
                results['routing_info'] = {
                    'rna_confidence': rna_confidence,
                    'threshold': self.routing.optimal_threshold,
                    'routed_to_multimodal': routing_used
                }
        
        # Add class names
        for key in results['predictions']:
            class_idx = results['predictions'][key]
            results['predictions'][key + '_name'] = self.config.CLASS_NAMES[class_idx]
        
        return results
    
    def predict_batch(self, csv_path, output_path=None):
        """
        Make predictions for a batch of samples from CSV
        
        Args:
            csv_path: Path to CSV with patient IDs and RNA data
            output_path: Path to save results
        
        Returns:
            results_df: DataFrame with all predictions
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if 'patient_id' not in df.columns:
            raise ValueError("CSV must contain 'patient_id' column")
        
        # Get gene columns
        gene_cols = [col for col in df.columns if col not in ['patient_id', 'PAM50']]
        
        all_results = []
        
        print(f"Processing {len(df)} samples...")
        
        for idx, row in df.iterrows():
            patient_id = row['patient_id']
            rna_data = row[gene_cols].values
            
            try:
                results = self.predict_single_sample(patient_id, rna_data)
                
                # Flatten results for DataFrame
                flat_results = {'patient_id': patient_id}
                
                # Add true label if available
                if 'PAM50' in row:
                    flat_results['true_label'] = row['PAM50']
                
                # Add predictions
                for model_name, pred in results['predictions'].items():
                    if not model_name.endswith('_name'):
                        flat_results[f'{model_name}_pred'] = pred
                        flat_results[f'{model_name}_pred_name'] = results['predictions'][f'{model_name}_name']
                        flat_results[f'{model_name}_conf'] = results['confidences'].get(model_name, None)
                
                all_results.append(flat_results)
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} samples")
                
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        # Calculate accuracy if true labels available
        if 'true_label' in results_df.columns:
            print("\nAccuracy Summary:")
            for col in results_df.columns:
                if col.endswith('_pred_name'):
                    model_name = col.replace('_pred_name', '')
                    accuracy = (results_df['true_label'] == results_df[col]).mean()
                    print(f"{model_name}: {accuracy:.4f}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Inference for breast cancer subtype classification'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch'],
        required=True,
        help='Inference mode: single sample or batch'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./outputs',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--patient_id',
        type=str,
        help='Patient ID for single sample inference'
    )
    parser.add_argument(
        '--rna_data',
        type=str,
        help='Path to RNA data (CSV or JSON) for single sample'
    )
    parser.add_argument(
        '--batch_csv',
        type=str,
        help='CSV file for batch inference'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./predictions.csv',
        help='Output path for predictions'
    )
    parser.add_argument(
        '--use_routing',
        action='store_true',
        help='Use routing mechanism'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine()
    engine.load_models(args.model_dir)
    
    # Load preprocessor
    preprocessor_path = os.path.join(args.model_dir, 'preprocessor.pkl')
    engine.load_preprocessor(preprocessor_path)
    
    if args.mode == 'single':
        if not args.patient_id or not args.rna_data:
            print("Error: --patient_id and --rna_data required for single mode")
            return
        
        # Load RNA data
        if args.rna_data.endswith('.csv'):
            rna_df = pd.read_csv(args.rna_data)
            rna_data = rna_df.iloc[0].values
        elif args.rna_data.endswith('.json'):
            with open(args.rna_data) as f:
                rna_data = json.load(f)
                rna_data = np.array(list(rna_data.values()))
        else:
            print("Error: RNA data must be CSV or JSON")
            return
        
        # Make prediction
        results = engine.predict_single_sample(
            args.patient_id,
            rna_data,
            use_routing=args.use_routing
        )
        
        # Print results
        print("\n" + "="*60)
        print(f"Prediction Results for Patient: {args.patient_id}")
        print("="*60)
        
        for model_name, pred_idx in results['predictions'].items():
            if not model_name.endswith('_name'):
                pred_name = results['predictions'][f'{model_name}_name']
                conf = results['confidences'].get(model_name, 'N/A')
                print(f"\n{model_name}:")
                print(f"  Predicted Class: {pred_name} (index: {pred_idx})")
                if conf != 'N/A':
                    print(f"  Confidence: {conf:.4f}")
        
        if 'routing_info' in results:
            print(f"\nRouting Information:")
            print(f"  RNA Confidence: {results['routing_info']['rna_confidence']:.4f}")
            print(f"  Threshold: {results['routing_info']['threshold']:.4f}")
            print(f"  Routed to Multimodal: {results['routing_info']['routed_to_multimodal']}")
        
        # Save to JSON
        output_json = args.output.replace('.csv', '.json')
        with open(output_json, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = results.copy()
            for key in results_serializable['probabilities']:
                results_serializable['probabilities'][key] = results_serializable['probabilities'][key].tolist()
            json.dump(results_serializable, f, indent=4)
        print(f"\nResults saved to {output_json}")
    
    elif args.mode == 'batch':
        if not args.batch_csv:
            print("Error: --batch_csv required for batch mode")
            return
        
        results_df = engine.predict_batch(args.batch_csv, args.output)
        print(f"\nBatch inference complete. Processed {len(results_df)} samples.")


if __name__ == "__main__":
    main()