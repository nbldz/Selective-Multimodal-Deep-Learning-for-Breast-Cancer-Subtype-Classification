"""
Attention Rollout for CTransPath Interpretability
Implements attention visualization as described in the paper
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image

class AttentionRollout:
    """
    Implements attention rollout to visualize which patches contribute 
    most to the model's predictions
    """
    
    def __init__(self, model, device, alpha=0.5):
        """
        Args:
            model: Trained model with transformer backbone
            device: torch device
            alpha: Weight for residual connections (0.5 = 50% attention, 50% residual)
        """
        self.model = model
        self.device = device
        self.alpha = alpha
        self.attention_maps = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention maps"""
        self.hooks = []
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Capture attention weights from multihead attention
                # Output format depends on implementation
                if hasattr(module, 'attn_weights'):
                    self.attention_maps.append(module.attn_weights.detach())
            return hook
        
        # Register hooks for all attention modules
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or isinstance(module, torch.nn.MultiheadAttention):
                handle = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_rollout(self, attention_matrices):
        """
        Compute attention rollout across layers
        
        Args:
            attention_matrices: List of attention matrices from each layer
                               Each matrix shape: (batch, num_heads, num_tokens, num_tokens)
        
        Returns:
            rollout: Final attention map (batch, num_tokens, num_tokens)
        """
        # Average across attention heads
        attention_avg = [attn.mean(dim=1) for attn in attention_matrices]
        
        # Add residual connections
        rollout = []
        for attn in attention_avg:
            batch_size, num_tokens, _ = attn.shape
            identity = torch.eye(num_tokens).to(attn.device).unsqueeze(0)
            # A_tilde = alpha * A + (1 - alpha) * I
            attn_residual = self.alpha * attn + (1 - self.alpha) * identity
            rollout.append(attn_residual)
        
        # Multiply attention matrices across layers
        result = rollout[0]
        for attn in rollout[1:]:
            result = torch.bmm(attn, result)
        
        return result
    
    def get_attention_map(self, wsi_patches, patch_counts=None):
        """
        Get attention map for WSI patches
        
        Args:
            wsi_patches: Tensor of shape (batch, num_patches, C, H, W)
            patch_counts: Number of valid patches per sample
        
        Returns:
            attention_weights: Attention weights for each patch
        """
        self.attention_maps = []
        self.register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            if hasattr(self.model, 'wsi_encoder'):
                # Multimodal model
                wsi_features = self.model.wsi_encoder(wsi_patches)
                if patch_counts is not None:
                    batch_size, max_patches = wsi_features.shape[0], wsi_features.shape[1]
                    mask = torch.arange(max_patches).expand(batch_size, max_patches).to(wsi_patches.device)
                    mask = mask < patch_counts.unsqueeze(1)
                    _, attention_weights = self.model.wsi_attention_pool(wsi_features, mask)
                else:
                    _, attention_weights = self.model.wsi_attention_pool(wsi_features)
            else:
                # WSI-only model
                features = self.model.feature_extractor(wsi_patches)
                if patch_counts is not None:
                    batch_size, max_patches = features.shape[0], features.shape[1]
                    mask = torch.arange(max_patches).expand(batch_size, max_patches).to(wsi_patches.device)
                    mask = mask < patch_counts.unsqueeze(1)
                    _, attention_weights = self.model.attention_pool(features, mask)
                else:
                    _, attention_weights = self.model.attention_pool(features)
        
        self.remove_hooks()
        
        return attention_weights
    
    def visualize_attention(
        self, 
        original_patches, 
        attention_weights, 
        patch_grid_shape=None,
        save_path=None,
        title="Attention Map"
    ):
        """
        Visualize attention weights on patches
        
        Args:
            original_patches: List of original patch images (PIL or numpy)
            attention_weights: Attention weights for each patch (numpy array)
            patch_grid_shape: Tuple (rows, cols) for arranging patches
            save_path: Path to save visualization
            title: Title for the plot
        """
        num_patches = len(original_patches)
        
        # Auto-determine grid shape if not provided
        if patch_grid_shape is None:
            grid_cols = int(np.ceil(np.sqrt(num_patches)))
            grid_rows = int(np.ceil(num_patches / grid_cols))
            patch_grid_shape = (grid_rows, grid_cols)
        
        # Normalize attention weights
        attn_norm = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min() + 1e-8
        )
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Attention weights bar chart
        ax1 = plt.subplot(2, 2, 1)
        bars = ax1.bar(range(num_patches), attention_weights)
        
        # Color bars by attention weight
        colors = plt.cm.RdYlGn(attn_norm)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel('Patch Index')
        ax1.set_ylabel('Attention Weight')
        ax1.set_title('Attention Weight Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap of attention in grid
        ax2 = plt.subplot(2, 2, 2)
        
        # Pad attention weights to fill grid
        grid_size = patch_grid_shape[0] * patch_grid_shape[1]
        attn_padded = np.pad(
            attn_norm[:num_patches], 
            (0, grid_size - num_patches),
            constant_values=0
        )
        attn_grid = attn_padded.reshape(patch_grid_shape)
        
        im = ax2.imshow(attn_grid, cmap='RdYlGn', aspect='auto')
        ax2.set_title('Attention Heatmap')
        ax2.set_xlabel('Grid Column')
        ax2.set_ylabel('Grid Row')
        plt.colorbar(im, ax=ax2)
        
        # Plot 3: Top attended patches
        ax3 = plt.subplot(2, 2, 3)
        top_k = min(5, num_patches)
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        for idx, patch_idx in enumerate(top_indices):
            if patch_idx < len(original_patches):
                patch = original_patches[patch_idx]
                if isinstance(patch, torch.Tensor):
                    patch = patch.permute(1, 2, 0).cpu().numpy()
                    # Denormalize if needed
                    patch = (patch * 0.229 + 0.485).clip(0, 1)
                
                # Create subplot for each top patch
                sub_ax = plt.subplot(2, top_k, top_k + idx + 1)
                sub_ax.imshow(patch)
                sub_ax.set_title(f'Patch {patch_idx}\nAttn: {attention_weights[patch_idx]:.3f}')
                sub_ax.axis('off')
        
        ax3.axis('off')
        ax3.set_title('Top 5 Attended Patches', y=0.95)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def create_overlay_heatmap(
        self, 
        wsi_image, 
        patch_positions, 
        attention_weights,
        patch_size=224,
        alpha=0.5,
        save_path=None
    ):
        """
        Create overlay heatmap on original WSI
        
        Args:
            wsi_image: Original WSI image (numpy array or PIL Image)
            patch_positions: List of (x, y) positions for each patch
            attention_weights: Attention weights for each patch
            patch_size: Size of each patch
            alpha: Transparency of overlay
            save_path: Path to save the overlay
        """
        if isinstance(wsi_image, Image.Image):
            wsi_image = np.array(wsi_image)
        
        # Create heatmap
        heatmap = np.zeros((wsi_image.shape[0], wsi_image.shape[1]))
        
        # Normalize attention weights
        attn_norm = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min() + 1e-8
        )
        
        # Fill heatmap at patch positions
        for (x, y), weight in zip(patch_positions, attn_norm):
            x_start, y_start = int(x), int(y)
            x_end = min(x_start + patch_size, heatmap.shape[1])
            y_end = min(y_start + patch_size, heatmap.shape[0])
            heatmap[y_start:y_end, x_start:x_end] = weight
        
        # Apply colormap
        cmap = plt.cm.jet
        heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB only
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend with original image
        overlay = cv2.addWeighted(
            wsi_image, 
            1 - alpha, 
            heatmap_colored, 
            alpha, 
            0
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(wsi_image)
        axes[0].set_title('Original WSI')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also save overlay separately
            overlay_path = save_path.replace('.png', '_overlay.png')
            Image.fromarray(overlay).save(overlay_path)
            print(f"Overlay saved to {save_path}")
        
        plt.show()
        
        return overlay


def visualize_predictions_with_attention(
    model,
    dataloader,
    device,
    num_samples=5,
    output_dir='./visualizations',
    class_names=None
):
    """
    Visualize model predictions with attention maps for multiple samples
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: torch device
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        class_names: List of class names
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    attention_rollout = AttentionRollout(model, device)
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            rna = batch['rna'].to(device)
            wsi = batch['wsi'].to(device)
            labels = batch['label']
            patch_counts = batch['patch_counts'].to(device)
            patient_ids = batch['patient_id']
            
            batch_size = rna.shape[0]
            
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                # Get single sample
                rna_sample = rna[i:i+1]
                wsi_sample = wsi[i:i+1]
                pc_sample = patch_counts[i:i+1]
                label = labels[i].item()
                patient_id = patient_ids[i]
                
                # Get prediction
                if hasattr(model, 'wsi_encoder'):  # Multimodal
                    logits, _, _ = model(rna_sample, wsi_sample, pc_sample)
                else:  # WSI only
                    logits, _, _ = model(wsi_sample, pc_sample)
                
                pred_probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = pred_probs[0, pred_class].item()
                
                # Get attention weights
                attention_weights = attention_rollout.get_attention_map(
                    wsi_sample, 
                    pc_sample
                )
                attention_weights = attention_weights[0].cpu().numpy()
                
                # Get valid patches
                num_valid_patches = pc_sample[0].item()
                valid_attention = attention_weights[:num_valid_patches]
                
                # Get original patches
                original_patches = []
                for j in range(num_valid_patches):
                    patch = wsi_sample[0, j].cpu()
                    original_patches.append(patch)
                
                # Create visualization
                true_label_name = class_names[label] if class_names else f"Class {label}"
                pred_label_name = class_names[pred_class] if class_names else f"Class {pred_class}"
                
                title = f"Patient: {patient_id}\nTrue: {true_label_name} | Pred: {pred_label_name} (conf: {confidence:.3f})"
                
                save_path = os.path.join(
                    output_dir, 
                    f"attention_{sample_count}_{patient_id}.png"
                )
                
                attention_rollout.visualize_attention(
                    original_patches,
                    valid_attention,
                    save_path=save_path,
                    title=title
                )
                
                sample_count += 1
                print(f"Visualized sample {sample_count}/{num_samples}")


def analyze_attention_patterns(
    model,
    dataloader,
    device,
    class_names=None,
    output_dir='./attention_analysis'
):
    """
    Analyze attention patterns across different classes
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: torch device
        class_names: List of class names
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    attention_rollout = AttentionRollout(model, device)
    
    # Store attention patterns per class
    class_attention = {i: [] for i in range(len(class_names) if class_names else 5)}
    
    model.eval()
    
    print("Analyzing attention patterns...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            rna = batch['rna'].to(device)
            wsi = batch['wsi'].to(device)
            labels = batch['label']
            patch_counts = batch['patch_counts'].to(device)
            
            # Get attention for each sample
            attention_weights = attention_rollout.get_attention_map(wsi, patch_counts)
            
            # Group by class
            for i, label in enumerate(labels):
                label = label.item()
                num_valid = patch_counts[i].item()
                attn = attention_weights[i, :num_valid].cpu().numpy()
                class_attention[label].append(attn)
    
    # Analyze and visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for class_idx, attention_list in class_attention.items():
        if not attention_list:
            continue
        
        ax = axes[class_idx]
        
        # Concatenate all attention weights for this class
        all_attention = np.concatenate(attention_list)
        
        # Plot distribution
        ax.hist(all_attention, bins=50, alpha=0.7, edgecolor='black')
        
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        ax.set_title(f'{class_name}\nMean: {all_attention.mean():.3f}, Std: {all_attention.std():.3f}')
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Weight Distributions by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'attention_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention analysis saved to {save_path}")
    plt.show()
    
    # Calculate statistics
    stats = {}
    for class_idx, attention_list in class_attention.items():
        if attention_list:
            all_attention = np.concatenate(attention_list)
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            stats[class_name] = {
                'mean': float(all_attention.mean()),
                'std': float(all_attention.std()),
                'min': float(all_attention.min()),
                'max': float(all_attention.max()),
                'median': float(np.median(all_attention))
            }
    
    # Save statistics
    import json
    stats_path = os.path.join(output_dir, 'attention_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Statistics saved to {stats_path}")
    
    return stats


if __name__ == "__main__":
    print("Attention Rollout module loaded successfully")
    print("Use this module to visualize attention patterns in your trained models")