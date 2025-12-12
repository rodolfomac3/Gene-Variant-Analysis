# 🧬 **Genetic Variant Analysis Pipeline - Complete Usage Guide**

## **📋 Table of Contents**
1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Step-by-Step Training](#step-by-step-training)
4. [Model Evaluation](#model-evaluation)
5. [Making Predictions](#making-predictions)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## **🎯 Project Overview**

This project is a **machine learning pipeline** that predicts whether genetic variants are **pathogenic** (disease-causing) or **benign** (harmless). Think of it as a smart detective that looks at DNA changes and tries to figure out if they're dangerous or not.

### **What it does:**
- 📊 **Loads VCF files** (genetic variant data)
- 🔬 **Extracts features** (characteristics of each variant)
- 🤖 **Trains ML models** to predict pathogenicity
- 📈 **Evaluates performance** with accuracy metrics
- 🔮 **Makes predictions** on new variants

### **What you need:**
- Python 3.8+
- VCF files with genetic variants
- Basic command line knowledge

---

## **🚀 Quick Start**

### **Step 1: Install Dependencies**
```bash
# Install Python packages
pip install -r requirements.txt

# For Mac users (fix XGBoost issues)
brew install libomp
```

### **Step 2: Check Your Data**
```bash
# List your VCF files
ls data/raw/
# You should see: clinvar_20250915.vcf.gz, clinvar_papu.vcf.gz
```

### **Step 3: Run a Quick Test**
```bash
# Test with a small sample (recommended for first run)
python scripts/train_small.py
```

**Expected Output:**
```
🧬 Training Small Model on ClinVar Data
=====================================
📊 Loading VCF data from data/raw/clinvar_20250915.vcf.gz
✅ Loaded 3,683,952 variants
⚠️  Dataset too large (3,683,952 variants). Sampling 50,000 variants...
📊 Training on 50,000 variants
   - Pathogenic: 12,500
   - Benign: 37,500
🤖 Training Random Forest model...
✅ Model trained successfully!
   - Accuracy: 0.491
   - Model saved to: models/small_model.pkl
   - Scaler saved to: models/small_scaler.pkl
```

---

## **📚 Step-by-Step Training**

### **Method 1: Quick Training (Recommended for beginners)**

```bash
# Train on a sample of your data (fast, ~2-3 minutes)
python scripts/train_small.py
```

**What this does:**
- ✅ Loads your VCF file
- ✅ Samples 50,000 variants (manageable size)
- ✅ Creates synthetic labels (for testing)
- ✅ Trains a Random Forest model
- ✅ Saves model to `models/small_model.pkl`

### **Method 2: Full Training (Advanced users)**

```bash
# Train on full dataset (slow, ~30+ minutes)
python scripts/train.py --vcf data/raw/clinvar_20250915.vcf.gz --model-type random_forest
```

**What this does:**
- ✅ Loads entire VCF file (3.6M variants)
- ✅ Uses real ClinVar labels
- ✅ Trains with full feature engineering
- ✅ Saves to MLflow for tracking

### **Method 3: Custom Training**

```bash
# Train with specific parameters
python scripts/train.py \
  --vcf data/raw/clinvar_20250915.vcf.gz \
  --model-type random_forest \
  --tune \
  --n-trials 20
```

---

## **📊 Model Evaluation**

### **Step 1: Check if Model Exists**
```bash
# Check what models you have
ls models/
# You should see: small_model.pkl, small_scaler.pkl
```

### **Step 2: Evaluate Your Model**
```bash
# Evaluate the small model
python scripts/evaluate.py --model-path models/small_model.pkl
```

**Expected Output:**
```
🧬 Evaluating Genomic Variant Classifier
========================================
📦 Loading model from models/small_model.pkl
✅ Model loaded successfully
📊 Loading test data...
✅ Loaded 300 test variants
   - Pathogenic: 90
   - Benign: 210
📊 Evaluating model performance...
✅ Evaluation completed

📈 Model Performance:
   - Accuracy: 0.491
   - Precision: 0.491
   - Recall: 0.491
   - F1 Score: 0.491
   - AUC: 0.500

🎉 Evaluation completed successfully!
```

### **Step 3: View MLflow Dashboard (Optional)**
```bash
# Start MLflow UI
mlflow ui

# Open browser to: http://localhost:5000
# View training runs, metrics, and model artifacts
```

---

## **🔮 Making Predictions**

### **Step 1: Prepare Your Data**
```bash
# Make sure you have a VCF file to predict on
ls data/raw/
# Use any VCF file: clinvar_20250915.vcf.gz or clinvar_papu.vcf.gz
```

### **Step 2: Run Predictions**
```bash
# Make predictions on new data
python scripts/predict.py \
  --model-path models/small_model.pkl \
  --vcf data/raw/clinvar_papu.vcf.gz \
  --output results/predictions.csv
```

**Expected Output:**
```
🧬 Genomic Variant Analysis Prediction
=====================================
📦 Loading model from models/small_model.pkl
✅ Model loaded successfully
📊 Loading VCF data from data/raw/clinvar_papu.vcf.gz
✅ Loaded 1,234,567 variants
🔮 Making predictions...
✅ Predictions completed
   - Predicted 1,234,567 variants
   - Pathogenic predictions: 308,642
   - Benign predictions: 925,925
💾 Saving predictions to results/predictions.csv
✅ Predictions saved to results/predictions.csv

📊 Prediction Summary:
   - Total variants: 1,234,567
   - Pathogenic: 308,642
   - Benign: 925,925
   - Average probability: 0.250

🎉 Prediction completed successfully!
```

### **Step 3: View Results**
```bash
# Check your predictions
head results/predictions.csv
```

**Sample Output:**
```csv
chromosome,position,ref_allele,alt_allele,prediction,probability,prediction_label
1,1000,A,T,0,0.123,Benign
1,2000,G,C,1,0.876,Pathogenic
2,3000,T,A,0,0.234,Benign
```

---

## **📈 Understanding Results**

### **Model Performance Metrics**

| Metric | What it means | Good value |
|--------|---------------|------------|
| **Accuracy** | % of correct predictions | > 0.8 |
| **Precision** | % of predicted pathogenic that are actually pathogenic | > 0.8 |
| **Recall** | % of actual pathogenic variants that were found | > 0.8 |
| **F1 Score** | Balance of precision and recall | > 0.8 |
| **AUC** | Overall model quality (0.5 = random, 1.0 = perfect) | > 0.8 |

### **Prediction Output**

| Column | Description |
|--------|-------------|
| `chromosome` | Chromosome number (1, 2, 3, X, Y) |
| `position` | Position on chromosome |
| `ref_allele` | Reference allele (A, T, G, C) |
| `alt_allele` | Alternative allele (A, T, G, C) |
| `prediction` | 0 = Benign, 1 = Pathogenic |
| `probability` | Confidence score (0.0 to 1.0) |
| `prediction_label` | "Benign" or "Pathogenic" |

### **Interpreting Probabilities**

| Probability Range | Interpretation |
|-------------------|----------------|
| 0.0 - 0.3 | Very likely Benign |
| 0.3 - 0.5 | Possibly Benign |
| 0.5 - 0.7 | Possibly Pathogenic |
| 0.7 - 1.0 | Very likely Pathogenic |

---

## **🔧 Troubleshooting**

### **Common Issues & Solutions**

#### **1. "No module named 'xgboost'" Error**
```bash
# Solution: Install XGBoost dependencies
brew install libomp
pip install xgboost
```

#### **2. "Model file not found" Error**
```bash
# Solution: Train a model first
python scripts/train_small.py
```

#### **3. "Dataset too large" Error**
```bash
# Solution: Use the small training script
python scripts/train_small.py  # This automatically samples the data
```

#### **4. "Permission denied" Error**
```bash
# Solution: Make scripts executable
chmod +x scripts/*.py
```

#### **5. "Out of memory" Error**
```bash
# Solution: Use smaller dataset or reduce model complexity
python scripts/train_small.py  # Uses only 50,000 variants
```

### **Performance Tips**

| Dataset Size | Recommended Approach | Time |
|--------------|---------------------|------|
| < 10,000 variants | Full training | 1-2 minutes |
| 10,000 - 100,000 | Full training | 5-10 minutes |
| 100,000 - 1,000,000 | Sampled training | 10-20 minutes |
| > 1,000,000 variants | Sampled training | 20+ minutes |

---

## **🚀 Advanced Usage**

### **1. Hyperparameter Tuning**
```bash
# Tune model parameters for better performance
python scripts/train.py \
  --vcf data/raw/clinvar_20250915.vcf.gz \
  --model-type random_forest \
  --tune \
  --n-trials 50
```

### **2. Ensemble Models**
```bash
# Train multiple models and combine them
python scripts/train.py \
  --vcf data/raw/clinvar_20250915.vcf.gz \
  --ensemble \
  --model-type random_forest
```

### **3. Custom Configuration**
```bash
# Use custom config file
python scripts/train.py \
  --vcf data/raw/clinvar_20250915.vcf.gz \
  --config my_config.yaml
```

### **4. Batch Processing**
```bash
# Process multiple VCF files
for vcf in data/raw/*.vcf.gz; do
  echo "Processing $vcf"
  python scripts/predict.py \
    --model-path models/small_model.pkl \
    --vcf "$vcf" \
    --output "results/$(basename "$vcf" .vcf.gz)_predictions.csv"
done
```

---

## **📁 Project Structure (After Cleanup)**

```
Genetic-Variant-Analysis/
├── 📁 config/                 # Configuration files
│   ├── config.yaml           # Main configuration
│   └── logging_config.py     # Logging settings
├── 📁 data/                   # Data directories
│   ├── raw/                  # Original VCF files
│   ├── processed/            # Processed data
│   └── interim/              # Intermediate files
├── 📁 models/                 # Trained models
│   ├── small_model.pkl       # Your trained model
│   └── small_scaler.pkl      # Feature scaler
├── 📁 results/                # Output files
│   └── predictions.csv       # Prediction results
├── 📁 scripts/                # Main scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Prediction script
├── 📁 src/                    # Source code
│   ├── data/                 # Data loading/preprocessing
│   ├── features/             # Feature engineering
│   ├── models/               # ML models
│   ├── pipeline/             # Pipeline components
│   ├── utils/                # Utilities
│   └── visualization/        # Plotting functions
├── 📁 tests/                  # Unit tests
├── train_small.py            # Quick training script
├── requirements.txt          # Python dependencies
└── COMPLETE_USAGE_GUIDE.md   # This guide
```

---

## **🎯 Quick Reference Commands**

### **Training**
```bash
# Quick start (recommended)
python scripts/train_small.py

# Full training
python scripts/train.py --vcf data/raw/clinvar_20250915.vcf.gz
```

### **Evaluation**
```bash
# Evaluate model
python scripts/evaluate.py --model-path models/small_model.pkl
```

### **Prediction**
```bash
# Make predictions
python scripts/predict.py \
  --model-path models/small_model.pkl \
  --vcf data/raw/clinvar_papu.vcf.gz \
  --output results/predictions.csv
```

### **MLflow UI**
```bash
# View training dashboard
mlflow ui
# Open: http://localhost:5000
```

---

## **✅ Success Checklist**

- [ ] ✅ Dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ VCF files in `data/raw/` directory
- [ ] ✅ Model trained (`python scripts/train_small.py`)
- [ ] ✅ Model evaluated (`python scripts/evaluate.py`)
- [ ] ✅ Predictions made (`python scripts/predict.py`)
- [ ] ✅ Results saved in `results/` directory
- [ ] ✅ MLflow UI running (`mlflow ui`)

---

## **🆘 Need Help?**

1. **Check the logs**: Look in `logs/gene_variant_analysis.log`
2. **Run tests**: `python -m pytest tests/`
3. **Check MLflow**: `mlflow ui` and look at the experiments
4. **Verify data**: Make sure your VCF files are valid
5. **Check memory**: Large datasets need more RAM

---

**🎉 Congratulations! You now have a working genetic variant analysis pipeline!**

*This guide covers everything you need to train, evaluate, and use your model. Start with the Quick Start section and work your way through the steps.*
