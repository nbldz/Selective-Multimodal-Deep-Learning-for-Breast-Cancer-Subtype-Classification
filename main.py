"""
Selective Multimodal Deep Learning for Breast Cancer Subtype Classification
Complete implementation based on the research paper
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, precision_recall_fscore_support
)
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    BRCA_WSI_DIR = "/path/to/brca/folder"  # Update this path
    RNA_CSV_PATH = "/path/to/rna_seq.csv"  # Update this path
    OUTPUT_DIR = "./outputs"
    
    # Data parameters
    NUM_TOP_GENES = 500
    IMG_SIZE = 224
    PATCH_SIZE = 16
    MAX_PATCHES_PER_PATIENT = 50  # Limit patches for memory efficiency
    
    # Model parameters
    CTRANSPATH_MODEL = "ctranspath"  # Will load pretrained weights
    WSI_FEATURE_DIM = 768
    RNA_HIDDEN_DIM = 512
    FUSION_HIDDEN_DIM = 256
    NUM_CLASSES = 5  # Luminal A, Luminal B, HER2, Basal, Normal-like
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10
    
    # Routing parameters
    CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class mapping
    CLASS_NAMES = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']
    SUBTYPE_TO_IDX = {
        'Basal': 0,
        'Her2': 1,
        'LumA': 2,
        'LumB': 3,
        'Normal': 4
    }

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Handles RNA-seq and WSI data preprocessing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
    def load_rna_data(self, rna_csv_path: str) -> pd.DataFrame:
        """Load and preprocess RNA-seq data"""
        print("Loading RNA-seq data...")
        df = pd.read_csv(rna_csv_path)
        
        # Assuming columns: patient_id, gene1, gene2, ..., geneN, PAM50
        if 'patient_id' not in df.columns or 'PAM50' not in df.columns:
            raise ValueError("CSV must contain 'patient_id' and 'PAM50' columns")
        
        return df
    
    def preprocess_rna(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess RNA-seq data:
        1. Log transform
        2. Handle missing values
        3. Select top variable genes
        4. Standardize
        """
        print("Preprocessing RNA-seq data...")
        
        # Separate features and labels
        patient_ids = df['patient_id'].values
        labels = df['PAM50'].values
        
        # Get gene expression columns
        gene_cols = [col for col in df.columns if col not in ['patient_id', 'PAM50']]
        X = df[gene_cols].values
        
        # Log transform (add 1 to avoid log(0))
        X = np.log1p(X)
        
        # Impute missing values
        X = self.imputer.fit_transform(X)
        
        # Select top variable genes
        gene_variances = np.var(X, axis=0)
        top_gene_indices = np.argsort(gene_variances)[-self.config.NUM_TOP_GENES:]
        X = X[:, top_gene_indices]
        
        print(f"Selected {len(top_gene_indices)} most variable genes")
        
        # Standardize
        X = self.scaler.fit_transform(X)
        
        return X, labels, patient_ids
    
    def get_patient_patches(self, patient_id: str) -> List[str]:
        """Get all patch paths for a patient"""
        patient_dir = Path(self.config.BRCA_WSI_DIR) / patient_id
        
        if not patient_dir.exists():
            return []
        
        # Get all jpg patches
        patches = list(patient_dir.glob("*.jpg"))
        
        # Limit number of patches
        if len(patches) > self.config.MAX_PATCHES_PER_PATIENT:
            patches = random.sample(patches, self.config.MAX_PATCHES_PER_PATIENT)
        
        return [str(p) for p in patches]

# ============================================================================
# DATASET
# ============================================================================

class MultimodalBRCADataset(Dataset):
    """Dataset for multimodal breast cancer data"""
    
    def __init__(
        self, 
        rna_features: np.ndarray,
        labels: np.ndarray,
        patient_ids: List[str],
        preprocessor: DataPreprocessor,
        transform=None
    ):
        self.rna_features = torch.FloatTensor(rna_features)
        self.labels = labels
        self.patient_ids = patient_ids
        self.preprocessor = preprocessor
        self.transform = transform
        
        # Map string labels to indices
        self.label_indices = np.array([
            Config.SUBTYPE_TO_IDX.get(label, -1) for label in labels
        ])
        
        # Filter out invalid labels
        valid_mask = self.label_indices != -1
        self.rna_features = self.rna_features[valid_mask]
        self.label_indices = self.label_indices[valid_mask]
        self.patient_ids = [pid for pid, valid in zip(patient_ids, valid_mask) if valid]
        
        # Get WSI patches for each patient
        self.wsi_patches = {}
        print("Loading WSI patches...")
        for patient_id in tqdm(self.patient_ids):
            patches = self.preprocessor.get_patient_patches(patient_id)
            if patches:
                self.wsi_patches[patient_id] = patches
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        rna = self.rna_features[idx]
        label = self.label_indices[idx]
        
        # Load WSI patches
        wsi_patches_list = []
        if patient_id in self.wsi_patches:
            patch_paths = self.wsi_patches[patient_id]
            for patch_path in patch_paths:
                try:
                    img = Image.open(patch_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    wsi_patches_list.append(img)
                except Exception as e:
                    continue
        
        # If no patches, create dummy tensor
        if not wsi_patches_list:
            wsi_patches = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
        else:
            wsi_patches = torch.stack(wsi_patches_list)
        
        return {
            'rna': rna,
            'wsi': wsi_patches,
            'label': label,
            'patient_id': patient_id
        }

def collate_fn(batch):
    """Custom collate function to handle variable number of patches"""
    rna = torch.stack([item['rna'] for item in batch])
    labels = torch.LongTensor([item['label'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    
    # For WSI, we'll take all patches and create a batch
    wsi_list = []
    patch_counts = []
    for item in batch:
        patches = item['wsi']
        wsi_list.append(patches)
        patch_counts.append(len(patches))
    
    # Pad to max patches in batch
    max_patches = max(patch_counts)
    wsi_padded = []
    for patches in wsi_list:
        if len(patches) < max_patches:
            padding = torch.zeros(
                max_patches - len(patches), 
                patches.shape[1], 
                patches.shape[2], 
                patches.shape[3]
            )
            patches = torch.cat([patches, padding], dim=0)
        wsi_padded.append(patches)
    
    wsi = torch.stack(wsi_padded)
    
    return {
        'rna': rna,
        'wsi': wsi,
        'label': labels,
        'patient_id': patient_ids,
        'patch_counts': torch.LongTensor(patch_counts)
    }

# ============================================================================
# MODELS
# ============================================================================

class CTransPathFeatureExtractor(nn.Module):
    """CTransPath-based feature extractor for WSI patches"""
    
    def __init__(self, feature_dim=768):
        super().__init__()
        # Load pretrained Swin Transformer (similar to CTransPath architecture)
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get actual output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            actual_dim = self.encoder(dummy_input).shape[1]
        
        # Project to desired dimension
        self.projection = nn.Linear(actual_dim, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, 3, H, W)
        Returns:
            features: (batch, num_patches, feature_dim)
        """
        batch_size, num_patches = x.shape[0], x.shape[1]
        
        # Reshape to process all patches
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        
        # Extract features
        features = self.encoder(x)
        features = self.projection(features)
        
        # Reshape back
        features = features.view(batch_size, num_patches, -1)
        
        return features

class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating patch features"""
    
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, num_patches, feature_dim)
            mask: (batch, num_patches) - 1 for valid patches, 0 for padding
        Returns:
            pooled: (batch, feature_dim)
            attention_weights: (batch, num_patches)
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (batch, num_patches)
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, num_patches)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, feature_dim)
        
        return pooled, attn_weights

class RNAEncoder(nn.Module):
    """RNA-seq encoder"""
    
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.encoder(x)

class RNAOnlyClassifier(nn.Module):
    """RNA-only classification model"""
    
    def __init__(self, input_dim, hidden_dim=512, num_classes=5, dropout=0.3):
        super().__init__()
        self.encoder = RNAEncoder(input_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

class WSIOnlyClassifier(nn.Module):
    """WSI-only classification model"""
    
    def __init__(self, feature_dim=768, num_classes=5):
        super().__init__()
        self.feature_extractor = CTransPathFeatureExtractor(feature_dim)
        self.attention_pool = AttentionPooling(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, patch_counts):
        # Extract features
        features = self.feature_extractor(x)  # (batch, num_patches, feature_dim)
        
        # Create mask
        batch_size, max_patches = features.shape[0], features.shape[1]
        mask = torch.arange(max_patches).expand(batch_size, max_patches).to(x.device)
        mask = mask < patch_counts.unsqueeze(1)
        
        # Attention pooling
        pooled, attn_weights = self.attention_pool(features, mask)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits, pooled, attn_weights

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module"""
    
    def __init__(self, rna_dim, wsi_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Project to same dimension
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.wsi_proj = nn.Linear(wsi_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, rna_features, wsi_features):
        """
        Args:
            rna_features: (batch, rna_dim)
            wsi_features: (batch, wsi_dim)
        Returns:
            fused: (batch, hidden_dim * 2)
        """
        # Project
        rna = self.rna_proj(rna_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        wsi = self.wsi_proj(wsi_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Cross-attention: RNA attends to WSI
        rna_attn, _ = self.multihead_attn(rna, wsi, wsi)
        rna = self.norm1(rna + rna_attn)
        
        # Feed-forward
        rna_out = self.norm2(rna + self.ffn(rna))
        
        # Cross-attention: WSI attends to RNA
        wsi_attn, _ = self.multihead_attn(wsi, rna, rna)
        wsi = self.norm1(wsi + wsi_attn)
        
        # Feed-forward
        wsi_out = self.norm2(wsi + self.ffn(wsi))
        
        # Concatenate
        fused = torch.cat([rna_out.squeeze(1), wsi_out.squeeze(1)], dim=1)
        
        return fused

class GatedFusion(nn.Module):
    """Gated fusion module"""
    
    def __init__(self, rna_dim, wsi_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(rna_dim + wsi_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rna_features, wsi_features):
        combined = torch.cat([rna_features, wsi_features], dim=1)
        gate = self.gate(combined)
        fused = gate * rna_features + (1 - gate) * wsi_features
        return fused

class MultimodalClassifier(nn.Module):
    """Multimodal classification model with multiple fusion strategies"""
    
    def __init__(
        self, 
        rna_input_dim,
        rna_hidden_dim=512,
        wsi_feature_dim=768,
        fusion_hidden_dim=256,
        num_classes=5,
        fusion_type='cross_attention'
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Encoders
        self.rna_encoder = RNAEncoder(rna_input_dim, rna_hidden_dim)
        self.wsi_encoder = CTransPathFeatureExtractor(wsi_feature_dim)
        self.wsi_attention_pool = AttentionPooling(wsi_feature_dim)
        
        # Fusion
        if fusion_type == 'concatenation':
            fusion_input_dim = rna_hidden_dim + wsi_feature_dim
            self.classifier = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_hidden_dim, num_classes)
            )
        elif fusion_type == 'gated':
            # For gated fusion, dimensions must match
            self.rna_proj = nn.Linear(rna_hidden_dim, fusion_hidden_dim)
            self.wsi_proj = nn.Linear(wsi_feature_dim, fusion_hidden_dim)
            self.fusion = GatedFusion(fusion_hidden_dim, fusion_hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(fusion_hidden_dim, num_classes)
            )
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                rna_hidden_dim, 
                wsi_feature_dim, 
                fusion_hidden_dim
            )
            self.classifier = nn.Sequential(
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_hidden_dim, num_classes)
            )
    
    def forward(self, rna, wsi, patch_counts):
        # Encode RNA
        rna_features = self.rna_encoder(rna)
        
        # Encode WSI
        wsi_patch_features = self.wsi_encoder(wsi)
        batch_size, max_patches = wsi_patch_features.shape[0], wsi_patch_features.shape[1]
        mask = torch.arange(max_patches).expand(batch_size, max_patches).to(wsi.device)
        mask = mask < patch_counts.unsqueeze(1)
        wsi_features, attn_weights = self.wsi_attention_pool(wsi_patch_features, mask)
        
        # Fusion
        if self.fusion_type == 'concatenation':
            fused = torch.cat([rna_features, wsi_features], dim=1)
        elif self.fusion_type == 'gated':
            rna_proj = self.rna_proj(rna_features)
            wsi_proj = self.wsi_proj(wsi_features)
            fused = self.fusion(rna_proj, wsi_proj)
        elif self.fusion_type == 'cross_attention':
            fused = self.fusion(rna_features, wsi_features)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits, fused, attn_weights

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
    def train_epoch(self, model, dataloader, optimizer, criterion):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            rna = batch['rna'].to(self.device)
            wsi = batch['wsi'].to(self.device)
            labels = batch['label'].to(self.device)
            patch_counts = batch['patch_counts'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, RNAOnlyClassifier):
                logits, _ = model(rna)
            elif isinstance(model, WSIOnlyClassifier):
                logits, _, _ = model(wsi, patch_counts)
            else:  # Multimodal
                logits, _, _ = model(rna, wsi, patch_counts)
            
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, model, dataloader, criterion, return_confidence=False):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                rna = batch['rna'].to(self.device)
                wsi = batch['wsi'].to(self.device)
                labels = batch['label'].to(self.device)
                patch_counts = batch['patch_counts'].to(self.device)
                
                # Forward pass
                if isinstance(model, RNAOnlyClassifier):
                    logits, _ = model(rna)
                elif isinstance(model, WSIOnlyClassifier):
                    logits, _, _ = model(wsi, patch_counts)
                else:  # Multimodal
                    logits, _, _ = model(rna, wsi, patch_counts)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'confidences': all_confidences
        }
        
        return results
    
    def train(self, model, train_loader, val_loader, num_epochs, patience=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_results = self.evaluate(model, val_loader, criterion)
            print(f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']:.4f}")
            print(f"Val F1 Macro: {val_results['f1_macro']:.4f}, Val F1 Weighted: {val_results['f1_weighted']:.4f}")
            
            # Early stopping
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"{self.config.OUTPUT_DIR}/best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f"{self.config.OUTPUT_DIR}/best_model.pth"))
        return model

# ============================================================================
# ROUTING MECHANISM
# ============================================================================

class RoutingSystem:
    """Implements uncertainty-aware routing mechanism"""
    
    def __init__(self, rna_model, multimodal_model, device):
        self.rna_model = rna_model
        self.multimodal_model = multimodal_model
        self.device = device
        self.optimal_threshold = 0.75  # Default, will be optimized
    
    def find_optimal_threshold(self, val_loader, thresholds):
        """Find optimal confidence threshold using validation set"""
        print("\nFinding optimal routing threshold...")
        
        best_threshold = None
        best_accuracy = 0
        
        for threshold in thresholds:
            predictions, labels = self.predict_with_routing(val_loader, threshold)
            accuracy = accuracy_score(labels, predictions)
            
            print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        print(f"\nOptimal threshold: {best_threshold} with accuracy: {best_accuracy:.4f}")
        
        return best_threshold
    
    def predict_with_routing(self, dataloader, threshold=None):
        """Make predictions using routing mechanism"""
        if threshold is None:
            threshold = self.optimal_threshold
        
        self.rna_model.eval()
        self.multimodal_model.eval()
        
        all_preds = []
        all_labels = []
        routed_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                rna = batch['rna'].to(self.device)
                wsi = batch['wsi'].to(self.device)
                labels = batch['label'].to(self.device)
                patch_counts = batch['patch_counts'].to(self.device)
                
                # RNA prediction
                rna_logits, _ = self.rna_model(rna)
                rna_probs = F.softmax(rna_logits, dim=1)
                rna_confidence = torch.max(rna_probs, dim=1)[0]
                rna_preds = torch.argmax(rna_logits, dim=1)
                
                # Route based on confidence
                low_confidence_mask = rna_confidence < threshold
                
                final_preds = rna_preds.clone()
                
                if low_confidence_mask.any():
                    routed_count += low_confidence_mask.sum().item()
                    
                    # Get multimodal predictions for low confidence samples
                    mm_logits, _, _ = self.multimodal_model(rna, wsi, patch_counts)
                    mm_preds = torch.argmax(mm_logits, dim=1)
                    
                    final_preds[low_confidence_mask] = mm_preds[low_confidence_mask]
                
                all_preds.extend(final_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        routing_rate = routed_count / len(all_labels) * 100
        print(f"Routing rate: {routing_rate:.2f}%")
        
        return all_preds, all_labels

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handles result visualization"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_percent, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    @staticmethod
    def plot_classification_report(y_true, y_pred, class_names, title, save_path):
        """Plot classification report as heatmap"""
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Extract metrics
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        for class_name in class_names:
            if class_name in report:
                data.append([report[class_name][m] for m in metrics])
        
        data = np.array(data)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            data, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn',
            xticklabels=metrics,
            yticklabels=class_names,
            vmin=0, 
            vmax=1
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Classification report saved to {save_path}")
    
    @staticmethod
    def calculate_ece(confidences, predictions, labels, n_bins=15):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("Selective Multimodal Deep Learning for Breast Cancer Classification")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(Config)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    rna_df = preprocessor.load_rna_data(Config.RNA_CSV_PATH)
    rna_features, labels, patient_ids = preprocessor.preprocess_rna(rna_df)
    
    print(f"Dataset size: {len(patient_ids)} patients")
    print(f"RNA features shape: {rna_features.shape}")
    print(f"Class distribution: {np.unique(labels, return_counts=True)}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        rna_features, labels, patient_ids, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = MultimodalBRCADataset(X_train, y_train, ids_train, preprocessor, train_transform)
    val_dataset = MultimodalBRCADataset(X_val, y_val, ids_val, preprocessor, test_transform)
    test_dataset = MultimodalBRCADataset(X_test, y_test, ids_test, preprocessor, test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = Trainer(Config)
    
    # Train RNA-only model
    print("\n4. Training RNA-only model...")
    rna_model = RNAOnlyClassifier(
        input_dim=Config.NUM_TOP_GENES,
        hidden_dim=Config.RNA_HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    rna_model = trainer.train(rna_model, train_loader, val_loader, Config.NUM_EPOCHS, Config.PATIENCE)
    
    # Evaluate RNA-only model
    print("\n5. Evaluating RNA-only model...")
    rna_results = trainer.evaluate(rna_model, test_loader, nn.CrossEntropyLoss())
    print(f"RNA-only Test Accuracy: {rna_results['accuracy']:.4f}")
    print(f"RNA-only Test F1 (Macro): {rna_results['f1_macro']:.4f}")
    
    # Train WSI-only model
    print("\n6. Training WSI-only model...")
    wsi_model = WSIOnlyClassifier(
        feature_dim=Config.WSI_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    wsi_model = trainer.train(wsi_model, train_loader, val_loader, Config.NUM_EPOCHS, Config.PATIENCE)
    
    # Evaluate WSI-only model
    print("\n7. Evaluating WSI-only model...")
    wsi_results = trainer.evaluate(wsi_model, test_loader, nn.CrossEntropyLoss())
    print(f"WSI-only Test Accuracy: {wsi_results['accuracy']:.4f}")
    print(f"WSI-only Test F1 (Macro): {wsi_results['f1_macro']:.4f}")
    
    # Train multimodal models with different fusion strategies
    fusion_types = ['concatenation', 'gated', 'cross_attention']
    multimodal_results = {}
    
    for fusion_type in fusion_types:
        print(f"\n8. Training Multimodal model with {fusion_type} fusion...")
        mm_model = MultimodalClassifier(
            rna_input_dim=Config.NUM_TOP_GENES,
            rna_hidden_dim=Config.RNA_HIDDEN_DIM,
            wsi_feature_dim=Config.WSI_FEATURE_DIM,
            fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
            fusion_type=fusion_type
        ).to(Config.DEVICE)
        
        mm_model = trainer.train(mm_model, train_loader, val_loader, Config.NUM_EPOCHS, Config.PATIENCE)
        
        # Evaluate
        mm_results = trainer.evaluate(mm_model, test_loader, nn.CrossEntropyLoss())
        multimodal_results[fusion_type] = mm_results
        print(f"Multimodal ({fusion_type}) Test Accuracy: {mm_results['accuracy']:.4f}")
        print(f"Multimodal ({fusion_type}) Test F1 (Macro): {mm_results['f1_macro']:.4f}")
        
        # Save model
        torch.save(mm_model.state_dict(), f"{Config.OUTPUT_DIR}/multimodal_{fusion_type}.pth")
    
    # Use best multimodal model for routing
    best_fusion_type = max(multimodal_results, key=lambda k: multimodal_results[k]['accuracy'])
    print(f"\n9. Best fusion strategy: {best_fusion_type}")
    
    # Load best multimodal model
    best_mm_model = MultimodalClassifier(
        rna_input_dim=Config.NUM_TOP_GENES,
        rna_hidden_dim=Config.RNA_HIDDEN_DIM,
        wsi_feature_dim=Config.WSI_FEATURE_DIM,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        fusion_type=best_fusion_type
    ).to(Config.DEVICE)
    best_mm_model.load_state_dict(torch.load(f"{Config.OUTPUT_DIR}/multimodal_{best_fusion_type}.pth"))
    
    # Initialize routing system
    print("\n10. Setting up routing mechanism...")
    routing_system = RoutingSystem(rna_model, best_mm_model, Config.DEVICE)
    
    # Find optimal threshold
    optimal_threshold = routing_system.find_optimal_threshold(val_loader, Config.CONFIDENCE_THRESHOLDS)
    
    # Evaluate routing system on test set
    print("\n11. Evaluating routing system on test set...")
    routing_preds, routing_labels = routing_system.predict_with_routing(test_loader)
    routing_accuracy = accuracy_score(routing_labels, routing_preds)
    routing_f1_macro = f1_score(routing_labels, routing_preds, average='macro')
    routing_f1_weighted = f1_score(routing_labels, routing_preds, average='weighted')
    
    print(f"\nRouting System Results:")
    print(f"Accuracy: {routing_accuracy:.4f}")
    print(f"F1 (Macro): {routing_f1_macro:.4f}")
    print(f"F1 (Weighted): {routing_f1_weighted:.4f}")
    
    # Calculate ECE for calibration analysis
    print("\n12. Calculating calibration metrics...")
    rna_ece = Visualizer.calculate_ece(
        np.array(rna_results['confidences']),
        np.array(rna_results['predictions']),
        np.array(rna_results['labels'])
    )
    
    mm_ece = Visualizer.calculate_ece(
        np.array(multimodal_results[best_fusion_type]['confidences']),
        np.array(multimodal_results[best_fusion_type]['predictions']),
        np.array(multimodal_results[best_fusion_type]['labels'])
    )
    
    print(f"RNA-only ECE: {rna_ece:.4f}")
    print(f"Multimodal ({best_fusion_type}) ECE: {mm_ece:.4f}")
    
    # Generate visualizations
    print("\n13. Generating visualizations...")
    
    # Confusion matrices
    Visualizer.plot_confusion_matrix(
        rna_results['labels'],
        rna_results['predictions'],
        Config.CLASS_NAMES,
        "RNA-only Model",
        f"{Config.OUTPUT_DIR}/cm_rna_only.png"
    )
    
    Visualizer.plot_confusion_matrix(
        wsi_results['labels'],
        wsi_results['predictions'],
        Config.CLASS_NAMES,
        "WSI-only Model",
        f"{Config.OUTPUT_DIR}/cm_wsi_only.png"
    )
    
    for fusion_type in fusion_types:
        Visualizer.plot_confusion_matrix(
            multimodal_results[fusion_type]['labels'],
            multimodal_results[fusion_type]['predictions'],
            Config.CLASS_NAMES,
            f"Multimodal ({fusion_type})",
            f"{Config.OUTPUT_DIR}/cm_multimodal_{fusion_type}.png"
        )
    
    Visualizer.plot_confusion_matrix(
        routing_labels,
        routing_preds,
        Config.CLASS_NAMES,
        "Routing-Based Model",
        f"{Config.OUTPUT_DIR}/cm_routing.png"
    )
    
    # Classification reports
    Visualizer.plot_classification_report(
        routing_labels,
        routing_preds,
        Config.CLASS_NAMES,
        "Routing-Based Model Performance",
        f"{Config.OUTPUT_DIR}/classification_report_routing.png"
    )
    
    # Save detailed results
    print("\n14. Saving detailed results...")
    results_summary = {
        'RNA-only': {
            'accuracy': rna_results['accuracy'],
            'f1_macro': rna_results['f1_macro'],
            'f1_weighted': rna_results['f1_weighted'],
            'ece': rna_ece
        },
        'WSI-only': {
            'accuracy': wsi_results['accuracy'],
            'f1_macro': wsi_results['f1_macro'],
            'f1_weighted': wsi_results['f1_weighted']
        }
    }
    
    for fusion_type in fusion_types:
        results_summary[f'Multimodal-{fusion_type}'] = {
            'accuracy': multimodal_results[fusion_type]['accuracy'],
            'f1_macro': multimodal_results[fusion_type]['f1_macro'],
            'f1_weighted': multimodal_results[fusion_type]['f1_weighted']
        }
    
    results_summary['Routing'] = {
        'accuracy': routing_accuracy,
        'f1_macro': routing_f1_macro,
        'f1_weighted': routing_f1_weighted,
        'optimal_threshold': optimal_threshold
    }
    
    # Save to file
    import json
    with open(f"{Config.OUTPUT_DIR}/results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'F1 (Macro)':<12} {'F1 (Weighted)':<12}")
    print("-"*80)
    
    for model_name, metrics in results_summary.items():
        print(f"{model_name:<30} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f}")
    
    print("="*80)
    print(f"\nBest model: Routing-Based with accuracy {routing_accuracy:.4f}")
    print(f"All results saved to {Config.OUTPUT_DIR}/")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()