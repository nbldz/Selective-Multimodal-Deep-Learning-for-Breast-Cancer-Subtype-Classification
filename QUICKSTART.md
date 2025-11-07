# Quick Start Guide - BRCA Classification

Complete guide to get you up and running in 15 minutes.

##  Prerequisites Checklist

- [ ] P100 GPU instance or equivalent 
- [ ] Ubuntu 20.04/22.04
- [ ] Python 3.8+
- [ ] TCGA-BRCA dataset downloaded
- [ ] SSH access configured

##  Step-by-Step Setup

### Step 1: Connect to AWS Instance

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Check GPU
nvidia-smi
```

**Expected output:** Should show P100 GPU with ~23GB memory

### Step 2: Clone/Upload Code

```bash
# Create project directory
mkdir -p ~/brca_classification
cd ~/brca_classification

# Upload all files via SCP (from your local machine)
scp -i your-key.pem -r /local/path/* ubuntu@your-instance-ip:~/brca_classification/

# Or git clone if you have a repository
git clone your-repo-url .
```

### Step 3: Run Setup Script

```bash
# Make executable
chmod +x setup_aws.sh

# Run setup (takes ~10 minutes)
./setup_aws.sh

# Activate environment
source venv/bin/activate
```

### Step 4: Prepare Your Data

Your data structure should look like:

```
~/brca_classification/
├── data/
│   ├── brca/                    # WSI patches
│   │   ├── TCGA-E2-A10A/
│   │   │   ├── patch_001.jpg
│   │   │   └── ...
│   │   └── TCGA-*/
│   └── rna_seq.csv              # RNA data
```

**RNA CSV Format:**
```csv
patient_id,gene1,gene2,...,PAM50
TCGA-E2-A10A,5.23,3.45,...,LumA
```

### Step 5: Validate Data

```bash
# Run data validation
python data_preparation.py \
    --brca_dir ./data/brca \
    --rna_csv ./data/rna_seq.csv \
    --output_dir ./prepared_data \
    --create_splits

# Check the report
cat ./prepared_data/data_report.txt
```

**Expected output:**
- ✓ Total patients matched
- ✓ No major warnings
- ✓ Balanced class distribution

### Step 6: Update Configuration

Edit `main.py` lines 45-47:

```python
class Config:
    BRCA_WSI_DIR = "/home/ubuntu/brca_classification/data/brca"
    RNA_CSV_PATH = "/home/ubuntu/brca_classification/data/rna_seq.csv"
    OUTPUT_DIR = "./outputs"
```

Or use the filtered data:

```python
    RNA_CSV_PATH = "./prepared_data/filtered_data.csv"
```

### Step 7: Start Training

```bash
# Make training script executable
chmod +x run_training.sh

# Start training in tmux (recommended)
tmux new -s brca
./run_training.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t brca
```

### Step 8: Monitor Progress

**In a new terminal:**

```bash
# SSH to instance again
ssh -i your-key.pem ubuntu@your-instance-ip

# Activate environment
cd ~/brca_classification
source venv/bin/activate

# Monitor training
python monitor_training.py

# Or check logs
tail -f logs/training_*.log
```

### Step 9: Check Results (After Training)

```bash
# Analyze results
python analyze_results.py --report_dir ./report

# View summary
python analyze_results.py --summary

# Check outputs
ls -lh outputs/
```

### Step 10: Run Inference

```bash
# Single sample
python inference.py \
    --mode single \
    --model_dir ./outputs \
    --patient_id TCGA-E2-A10A \
    --rna_data ./data/sample_rna.json \
    --use_routing

# Batch inference
python inference.py \
    --mode batch \
    --batch_csv ./prepared_data/test.csv \
    --output ./predictions.csv \
    --use_routing
```

##  Timeline

| Step | Time | Task |
|------|------|------|
| 1-3 | 15 min | Setup environment |
| 4-5 | 10 min | Prepare and validate data |
| 6 | 2 min | Configure paths |
| 7 | 4-6 hrs | Training (can run overnight) |
| 8 | Ongoing | Monitoring |
| 9-10 | 5 min | Results and inference |

**Total hands-on time:** ~30 minutes  
**Total compute time:** 4-6 hours

##  Common Issues & Solutions

### Issue 1: CUDA Out of Memory

```python
# In main.py, reduce batch size
Config.BATCH_SIZE = 16  # or 8
Config.MAX_PATCHES_PER_PATIENT = 30
```

### Issue 2: Missing Dependencies

```bash
pip install -r requirements.txt
# Or individually:
pip install torch torchvision timm scikit-learn pandas matplotlib seaborn
```

### Issue 3: Data Not Found

```bash
# Check paths
python -c "from main import Config; c=Config(); print(c.BRCA_WSI_DIR); print(c.RNA_CSV_PATH)"

# Verify files exist
ls -la /path/to/brca/folder
ls -la /path/to/rna_seq.csv
```

### Issue 4: Patient ID Mismatch

```bash
# Run data validation
python data_preparation.py --brca_dir ./data/brca --rna_csv ./data/rna_seq.csv --validate_only

# Check alignment rate (should be >80%)
```

### Issue 5: Training Stalls

```bash
# Check GPU usage
nvidia-smi

# Check process
ps aux | grep python

# Check logs
tail -100 logs/training_*.log
```

##  Expected Results

After successful training, you should see:

```
FINAL RESULTS SUMMARY
================================================================================
Model                          Accuracy     F1 (Macro)   F1 (Weighted)
--------------------------------------------------------------------------------
RNA-only                       0.9100       0.9000       0.9100
WSI-only                       0.6200       0.5700       0.6400
Multimodal-concatenation       0.9350       0.9200       0.9300
Multimodal-gated              0.9420       0.9300       0.9400
Multimodal-cross_attention    0.9480       0.9300       0.9500
Routing                       0.9505       0.9300       0.9500
================================================================================
```

##  Output Files

After training completes:

```
outputs/
├── best_model.pth                      # RNA-only model (150MB)
├── multimodal_cross_attention.pth      # Best multimodal (400MB)
├── cm_routing.png                      # Confusion matrix
├── results_summary.json                # All metrics
└── preprocessor.pkl                    # For inference

logs/
└── training_20250122_143022.log       # Training log

report/
├── REPORT.md                          # Analysis report
├── model_comparison.png
└── calibration_comparison.png
```

##  Next Steps

1. **Review Results:** Check `report/REPORT.md`
2. **Analyze Attention:** Use `attention_rollout.py`
3. **Deploy Model:** Use `inference.py` for predictions
4. **Fine-tune:** Adjust hyperparameters in `config.yaml`
5. **Optimize:** Try different fusion strategies

##  Pro Tips

1. **Use tmux** for long training sessions
2. **Monitor GPU** with `nvidia-smi -l 1`
3. **Check logs** regularly during training
4. **Backup models** after successful training
5. **Validate data** before starting training

##  Getting Help

If you encounter issues:

1. Check this guide first
2. Review error messages in logs
3. Verify data format matches requirements
4. Check GPU memory with `nvidia-smi`
5. Ensure all dependencies are installed

##  Quick Commands Reference

```bash
# Setup
./setup_aws.sh && source venv/bin/activate

# Validate data
python data_preparation.py --brca_dir ./data/brca --rna_csv ./data/rna_seq.csv --validate_only

# Train
tmux new -s brca && ./run_training.sh

# Monitor
python monitor_training.py

# Analyze
python analyze_results.py

# Inference
python inference.py --mode batch --batch_csv test.csv --output predictions.csv
```

---

**Happy Training! **

For detailed documentation, see [README.md](README.md)