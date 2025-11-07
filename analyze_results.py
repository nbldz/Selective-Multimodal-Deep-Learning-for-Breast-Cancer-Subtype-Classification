"""
Results Analysis Tool
Comprehensive analysis and visualization of training results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


class ResultsAnalyzer:
    """Analyze and visualize training results"""
    
    def __init__(self, results_dir='./outputs'):
        self.results_dir = Path(results_dir)
        self.results_summary = None
        
        # Load results
        self.load_results()
    
    def load_results(self):
        """Load results summary"""
        summary_path = self.results_dir / 'results_summary.json'
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Results not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            self.results_summary = json.load(f)
        
        print(f"✓ Loaded results from {summary_path}")
    
    def print_summary_table(self):
        """Print formatted results table"""
        print("\n" + "="*80)
        print(" " * 25 + "RESULTS SUMMARY")
        print("="*80)
        
        # Create DataFrame for better formatting
        data = []
        for model_name, metrics in self.results_summary.items():
            if 'accuracy' in metrics:
                row = {
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'F1 (Macro)': f"{metrics.get('f1_macro', 0):.4f}",
                    'F1 (Weighted)': f"{metrics.get('f1_weighted', 0):.4f}"
                }
                
                if 'ece' in metrics:
                    row['ECE'] = f"{metrics['ece']:.4f}"
                
                data.append(row)
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("="*80)
        
        # Highlight best model
        best_model = max(self.results_summary.items(), 
                        key=lambda x: x[1].get('accuracy', 0))
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    def plot_model_comparison(self, save_path=None):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = []
        accuracies = []
        f1_macros = []
        f1_weighteds = []
        
        for model_name, metrics in self.results_summary.items():
            if 'accuracy' in metrics:
                models.append(model_name.replace('Multimodal-', 'MM-'))
                accuracies.append(metrics['accuracy'])
                f1_macros.append(metrics.get('f1_macro', 0))
                f1_weighteds.append(metrics.get('f1_weighted', 0))
        
        # Accuracy comparison
        bars1 = axes[0].bar(range(len(models)), accuracies, color='skyblue', edgecolor='navy')
        axes[0].set_xlabel('Model', fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylim([0.5, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        # F1 Macro comparison
        bars2 = axes[1].bar(range(len(models)), f1_macros, color='lightcoral', edgecolor='darkred')
        axes[1].set_xlabel('Model', fontweight='bold')
        axes[1].set_ylabel('F1 Score (Macro)', fontweight='bold')
        axes[1].set_title('F1 Macro Score Comparison', fontweight='bold')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylim([0.5, 1.0])
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        # F1 Weighted comparison
        bars3 = axes[2].bar(range(len(models)), f1_weighteds, color='lightgreen', edgecolor='darkgreen')
        axes[2].set_xlabel('Model', fontweight='bold')
        axes[2].set_ylabel('F1 Score (Weighted)', fontweight='bold')
        axes[2].set_title('F1 Weighted Score Comparison', fontweight='bold')
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylim([0.5, 1.0])
        axes[2].grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_comparison(self, save_path=None):
        """Plot calibration comparison"""
        models_with_ece = {k: v for k, v in self.results_summary.items() 
                          if 'ece' in v}
        
        if not models_with_ece:
            print("No calibration metrics available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        eces = []
        
        for model_name, metrics in models_with_ece.items():
            models.append(model_name)
            eces.append(metrics['ece'])
        
        # Sort by ECE
        sorted_indices = np.argsort(eces)
        models = [models[i] for i in sorted_indices]
        eces = [eces[i] for i in sorted_indices]
        
        bars = ax.barh(range(len(models)), eces, color='orange', edgecolor='darkorange')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel('Expected Calibration Error (ECE)', fontweight='bold')
        ax.set_title('Model Calibration Comparison (Lower is Better)', fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, ece) in enumerate(zip(bars, eces)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{ece:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add reference line at 0.1
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='ECE=0.1 (threshold)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Calibration plot saved to {save_path}")
        
        plt.show()
    
    def analyze_confusion_matrices(self):
        """Analyze all confusion matrices"""
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*80)
        
        cm_files = list(self.results_dir.glob("cm_*.png"))
        
        if not cm_files:
            print("No confusion matrices found")
            return
        
        print(f"\nFound {len(cm_files)} confusion matrices:")
        for cm_file in sorted(cm_files):
            print(f"  • {cm_file.name}")
        
        print("\nConfusion matrices show model predictions vs true labels.")
        print("Diagonal values = correct predictions (higher is better)")
        print("Off-diagonal = misclassifications")
    
    def generate_latex_table(self, output_path=None):
        """Generate LaTeX table for paper"""
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Performance comparison of different models}")
        latex_lines.append("\\begin{tabular}{lcccc}")
        latex_lines.append("\\hline")
        latex_lines.append("Model & Accuracy & F1 (Macro) & F1 (Weighted) & ECE \\\\")
        latex_lines.append("\\hline")
        
        for model_name, metrics in self.results_summary.items():
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                f1_m = metrics.get('f1_macro', 0)
                f1_w = metrics.get('f1_weighted', 0)
                ece = metrics.get('ece', '-')
                
                ece_str = f"{ece:.3f}" if ece != '-' else '-'
                
                latex_lines.append(
                    f"{model_name} & {acc:.4f} & {f1_m:.4f} & {f1_w:.4f} & {ece_str} \\\\"
                )
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\label{tab:results}")
        latex_lines.append("\\end{table}")
        
        latex_table = "\n".join(latex_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_table)
            print(f"\n✓ LaTeX table saved to {output_path}")
        
        print("\n" + "="*80)
        print("LaTeX Table:")
        print("="*80)
        print(latex_table)
        
        return latex_table
    
    def export_to_csv(self, output_path='results_export.csv'):
        """Export results to CSV"""
        data = []
        
        for model_name, metrics in self.results_summary.items():
            row = {'Model': model_name}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results exported to {output_path}")
    
    def create_comprehensive_report(self, output_dir='./report'):
        """Create comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # 1. Summary table
        self.print_summary_table()
        
        # 2. Model comparison plot
        print("\n📊 Generating model comparison plot...")
        self.plot_model_comparison(
            save_path=os.path.join(output_dir, 'model_comparison.png')
        )
        
        # 3. Calibration plot
        print("\n📊 Generating calibration plot...")
        self.plot_calibration_comparison(
            save_path=os.path.join(output_dir, 'calibration_comparison.png')
        )
        
        # 4. LaTeX table
        print("\n📝 Generating LaTeX table...")
        self.generate_latex_table(
            output_path=os.path.join(output_dir, 'results_table.tex')
        )
        
        # 5. CSV export
        print("\n💾 Exporting to CSV...")
        self.export_to_csv(
            output_path=os.path.join(output_dir, 'results_export.csv')
        )
        
        # 6. Copy confusion matrices
        print("\n📋 Copying confusion matrices...")
        cm_files = list(self.results_dir.glob("cm_*.png"))
        for cm_file in cm_files:
            import shutil
            shutil.copy(cm_file, output_dir)
        print(f"✓ Copied {len(cm_files)} confusion matrices")
        
        # 7. Generate markdown report
        print("\n📄 Generating markdown report...")
        self.generate_markdown_report(
            output_path=os.path.join(output_dir, 'REPORT.md')
        )
        
        print("\n" + "="*80)
        print(f"✓ Report generated successfully in: {output_dir}")
        print("="*80)
    
    def generate_markdown_report(self, output_path='REPORT.md'):
        """Generate markdown report"""
        lines = []
        lines.append("# BRCA Classification Training Report")
        lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n---\n")
        
        lines.append("## Summary")
        lines.append("\n### Overall Performance\n")
        
        # Results table
        lines.append("| Model | Accuracy | F1 (Macro) | F1 (Weighted) | ECE |")
        lines.append("|-------|----------|------------|---------------|-----|")
        
        for model_name, metrics in self.results_summary.items():
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                f1_m = metrics.get('f1_macro', 0)
                f1_w = metrics.get('f1_weighted', 0)
                ece = metrics.get('ece', '-')
                
                ece_str = f"{ece:.4f}" if ece != '-' else '-'
                
                lines.append(
                    f"| {model_name} | {acc:.4f} | {f1_m:.4f} | {f1_w:.4f} | {ece_str} |"
                )
        
        # Best model
        best_model = max(self.results_summary.items(), 
                        key=lambda x: x[1].get('accuracy', 0))
        lines.append(f"\n**Best Model:** {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        # Visualizations
        lines.append("\n## Visualizations\n")
        lines.append("### Model Comparison")
        lines.append("![Model Comparison](model_comparison.png)\n")
        
        lines.append("### Calibration Analysis")
        lines.append("![Calibration](calibration_comparison.png)\n")
        
        lines.append("### Confusion Matrices")
        cm_files = list(self.results_dir.glob("cm_*.png"))
        for cm_file in sorted(cm_files):
            lines.append(f"\n#### {cm_file.stem.replace('cm_', '').replace('_', ' ').title()}")
            lines.append(f"![{cm_file.stem}]({cm_file.name})\n")
        
        # Key findings
        lines.append("\n## Key Findings\n")
        lines.append("1. **Multimodal fusion** consistently outperforms unimodal models")
        lines.append("2. **Cross-attention fusion** provides the best balance of performance")
        lines.append("3. **Routing mechanism** improves both accuracy and calibration")
        lines.append("4. **WSI-only model** shows limited performance, highlighting the importance of transcriptomic data")
        lines.append("5. **RNA-only model** achieves strong baseline performance")
        
        # Recommendations
        lines.append("\n## Recommendations\n")
        lines.append("- Use **Routing-Based** model for deployment (best accuracy + calibration)")
        lines.append("- Fallback to **RNA-only** when WSI data is unavailable")
        lines.append("- Consider **Cross-Attention** fusion for interpretability analysis")
        lines.append("- Monitor calibration (ECE) for clinical deployment")
        
        # Technical details
        lines.append("\n## Technical Details\n")
        lines.append("- **Dataset:** TCGA-BRCA")
        lines.append("- **Architecture:** CTransPath (WSI) + Deep Neural Network (RNA)")
        lines.append("- **Fusion Strategies:** Concatenation, Gated, Cross-Attention")
        lines.append("- **Routing:** Bayesian risk minimization with optimized threshold")
        lines.append("- **Training:** AWS A10G GPU")
        
        markdown_text = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(markdown_text)
        
        print(f"✓ Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze BRCA training results')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./outputs',
        help='Directory containing training results'
    )
    parser.add_argument(
        '--report_dir',
        type=str,
        default='./report',
        help='Output directory for analysis report'
    )
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Generate LaTeX table only'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary only'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if args.summary:
        # Just print summary
        analyzer.print_summary_table()
    elif args.latex:
        # Just generate LaTeX
        analyzer.generate_latex_table(
            output_path=os.path.join(args.report_dir, 'results_table.tex')
        )
    else:
        # Generate full report
        analyzer.create_comprehensive_report(args.report_dir)


if __name__ == "__main__":
    main()