#  Quick Start Guide - Genomic Variant Analysis Pipeline

##  What We've Built

You now have a **complete, production-ready genomic variant analysis pipeline** with:

-  **Data Processing**: VCF parsing, annotation integration, data validation
-  **Feature Engineering**: Variant, sequence, and annotation features
-  **ML Models**: XGBoost, LightGBM, RandomForest, Ensemble methods
-  **MLOps**: MLflow tracking, DVC pipelines, Docker containerization
-  **Testing**: Unit tests, integration tests, CI/CD
-  **Visualization**: ROC curves, feature importance, data distributions

## How to Use Your Pipeline

### **1. Install Dependencies**

```bash
# Install core dependencies
pip install -r requirements.txt

# Fix XGBoost on macOS (if needed)
brew install libomp
pip install xgboost

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### **2. Test the Pipeline**

```bash
# Run the working test
python test_working.py

# Run the comprehensive demo
python demo_pipeline.py
```

### **3. Use with Your Data**

#### **Option A: Using the Scripts (Recommended)**

```bash
# 1. Add your VCF file to data/raw/
cp your_variants.vcf data/raw/

# 2. Train a model
python scripts/train.py --vcf data/raw/your_variants.vcf --model-type random_forest

# 3. Evaluate the model
python scripts/evaluate.py --model-path models/best_model.pkl

# 4. Make predictions on new data
python scripts/predict.py --input data/raw/new_variants.vcf --output results/predictions.csv
```

#### **Option B: Using Python Code**

```python
import pandas as pd
import numpy as np
from src.models.variant_classifier import VariantClassifier
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('data/raw/your_variants.csv')

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['pathogenicity', 'chromosome']]
X = df[feature_cols]
y = df['pathogenicity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = VariantClassifier(model_type='random_forest')
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### **4. View Results**

```bash
# Start MLflow UI to view experiments
mlflow ui

# Open in browser: http://localhost:5000
```

## 📁 Project Structure

```
Genetic-Variant-Analysis/
├──  data/                    # Your data goes here
│   ├── raw/                   # VCF files
│   ├── processed/             # Processed data
│   └── external/              # Annotations
├── 🔧 src/                    # Source code
│   ├── data/                  # Data loading & preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # ML models
│   ├── pipeline/              # Training/evaluation pipelines
│   ├── utils/                 # Utilities
│   └── visualization/         # Plotting
├──  scripts/                # Command-line scripts
├──  tests/                  # Unit tests
├──  config/                 # Configuration
├──  results/                # Output results
└──  Docker files            # Containerization
```

##  Available Commands

### **Using Makefile**

```bash
# Install dependencies
make install

# Run training
make train

# Run evaluation
make evaluate

# Run tests
make test

# Build Docker image
make docker-build

# Clean up
make clean
```

### **Using DVC Pipeline**

```bash
# Initialize DVC
dvc init

# Run complete pipeline
dvc repro

# View pipeline status
dvc status
```

##  Key Features

### **1. Multiple ML Models**
- XGBoost (with hyperparameter tuning)
- LightGBM
- Random Forest
- Gradient Boosting
- Logistic Regression
- Ensemble methods (Voting, Stacking)

### **2. Comprehensive Feature Engineering**
- Variant features (type, length, frequency)
- Sequence features (motifs, structure, GC content)
- Annotation features (consequences, pathways)
- Population genetics features

### **3. MLOps Integration**
- MLflow experiment tracking
- Model versioning and deployment
- DVC data versioning
- Docker containerization
- CI/CD with GitHub Actions

### **4. Production Ready**
- Comprehensive logging
- Error handling
- Configuration management
- Unit and integration tests
- Documentation

##  Troubleshooting

### **Common Issues**

1. **XGBoost Error on macOS**
   ```bash
   brew install libomp
   pip install xgboost
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Path Issues**
   - Ensure VCF files are in `data/raw/`
   - Check file permissions
   - Verify file format

### **Getting Help**

- Check logs in `logs/` directory
- Review MLflow UI for experiment details
- Run `python test_working.py` for basic functionality
- Check GitHub Issues for common problems

##  Success Indicators

 **Your pipeline is working when you see:**
- Test script runs without errors
- MLflow UI shows experiments
- Models are saved in `models/` directory
- Results are generated in `results/` directory
- Logs show successful completion

##  Next Steps

1. **Add Your Data**: Place VCF files in `data/raw/`
2. **Run Training**: `python scripts/train.py --vcf data/raw/your_data.vcf`
3. **View Results**: `mlflow ui`
4. **Make Predictions**: `python scripts/predict.py`
5. **Customize**: Modify `config/config.yaml` for your needs

##  Advanced Usage

### **Hyperparameter Tuning**
```bash
python scripts/train.py --vcf data/raw/your_data.vcf --tune --n-trials 100
```

### **Ensemble Models**
```bash
python scripts/train.py --vcf data/raw/your_data.vcf --ensemble
```

### **Docker Deployment**
```bash
docker build -t genomic-variant-analysis .
docker run -v $(pwd)/data:/app/data genomic-variant-analysis python scripts/train.py
```

---

** You now have a complete, production-ready genomic variant analysis pipeline!**

Start by running `python test_working.py` to verify everything works, then add your data and begin training models.
