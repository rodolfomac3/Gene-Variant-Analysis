# 🧬 Genomic Variant Analysis Pipeline - Usage Guide

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 2. **Test the Pipeline**
```bash
# Run the comprehensive test suite
python test_pipeline.py
```

### 3. **Run with Your Data**
```bash
# Place your VCF files in data/raw/
# Then run the training pipeline
python scripts/train.py

# Evaluate the trained model
python scripts/evaluate.py

# Make predictions on new data
python scripts/predict.py --input data/raw/new_variants.vcf --output results/predictions.csv
```

## 📁 Project Structure Overview

```
Genetic-Variant-Analysis/
├──  data/                    # Data storage
│   ├── raw/                   # Raw VCF files
│   ├── processed/             # Processed data
│   └── external/              # External annotations
├──  src/                    # Source code
│   ├── data/                  # Data loading & preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # ML models
│   ├── pipeline/              # Training/evaluation pipelines
│   ├── utils/                 # Utility functions
│   └── visualization/         # Plotting functions
├──  scripts/                # Command-line scripts
├── tests/                  # Unit tests
├──  config/                 # Configuration files
└──  results/                # Output results
```

##  How to Use Each Component

### **1. Data Loading & Preprocessing**
```python
from src.data.data_loader import GenomicDataLoader
from src.data.preprocessor import GenomicPreprocessor

# Load VCF data
loader = GenomicDataLoader()
variants = loader.load_vcf('data/raw/your_variants.vcf')

# Preprocess data
preprocessor = GenomicPreprocessor()
preprocessor.fit(variants)
processed_data = preprocessor.transform(variants)
```

### **2. Feature Engineering**
```python
from src.features.variant_features import VariantFeatureExtractor
from src.features.sequence_features import SequenceFeatureExtractor

# Extract variant features
variant_extractor = VariantFeatureExtractor()
variant_features = variant_extractor.extract_basic_features(variants)

# Extract sequence features
sequence_extractor = SequenceFeatureExtractor()
sequence_features = sequence_extractor.extract_motif_features(variants)
```

### **3. Model Training**
```python
from src.models.variant_classifier import VariantClassifier

# Train a model
model = VariantClassifier(model_type='xgboost')
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### **4. Full Pipeline**
```python
from src.pipeline.training_pipeline import TrainingPipeline

# Run complete training pipeline
pipeline = TrainingPipeline()
pipeline.run()
```

##  Available Scripts

### **Training Script** (`scripts/train.py`)
```bash
python scripts/train.py [options]

Options:
  --config CONFIG     Path to config file (default: config/config.yaml)
  --data-path PATH    Path to input data
  --model-type TYPE   Model type (xgboost, lightgbm, random_forest, etc.)
  --output-dir DIR    Output directory for results
```

### **Evaluation Script** (`scripts/evaluate.py`)
```bash
python scripts/evaluate.py [options]

Options:
  --model-path PATH   Path to trained model
  --test-data PATH    Path to test data
  --output-dir DIR    Output directory for results
```

### **Prediction Script** (`scripts/predict.py`)
```bash
python scripts/predict.py [options]

Options:
  --input PATH        Input VCF file
  --model-path PATH   Path to trained model
  --output PATH       Output CSV file
```

##  Docker Usage

### **Build and Run with Docker**
```bash
# Build the Docker image
docker build -t genomic-variant-analysis .

# Run the training pipeline
docker run -v $(pwd)/data:/app/data genomic-variant-analysis python scripts/train.py

# Run with Docker Compose
docker-compose up
```

##  MLflow Integration

### **View Experiments**
```bash
# Start MLflow UI
mlflow ui

# Open in browser: http://localhost:5000
```

### **Track Experiments**
- All training runs are automatically logged to MLflow
- View metrics, parameters, and artifacts
- Compare different model versions
- Deploy best models

##  DVC Pipeline

### **Run DVC Pipeline**
```bash
# Initialize DVC (if not already done)
dvc init

# Run the complete pipeline
dvc repro

# View pipeline status
dvc status
```

##  Testing

### **Run Tests**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src
```

### **Code Quality**
```bash
# Lint code
flake8 src/

# Format code
black src/

# Type checking
mypy src/
```

##  Monitoring & Logging

### **View Logs**
```bash
# Training logs
tail -f logs/training.log

# Application logs
tail -f logs/app.log
```

### **MLflow Tracking**
- **UI**: http://localhost:5000
- **API**: http://localhost:5000/api
- **Artifacts**: Stored in `mlruns/` directory

##  Production Deployment

### **Using Makefile**
```bash
# Install dependencies
make install

# Run training
make train

# Run evaluation
make evaluate

# Build Docker image
make docker-build

# Run tests
make test

# Clean up
make clean
```

##  Configuration

### **Main Config** (`config/config.yaml`)
- Data paths and parameters
- Model hyperparameters
- MLflow settings
- Feature engineering parameters

### **Environment Variables**
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export DATA_PATH="/path/to/your/data"
export MODEL_PATH="/path/to/your/models"
```

##  Example Workflow

1. **Prepare Data**
   ```bash
   # Place VCF files in data/raw/
   cp your_variants.vcf data/raw/
   ```

2. **Run Training**
   ```bash
   python scripts/train.py --data-path data/raw/your_variants.vcf
   ```

3. **Evaluate Model**
   ```bash
   python scripts/evaluate.py --model-path models/best_model.pkl
   ```

4. **Make Predictions**
   ```bash
   python scripts/predict.py --input data/raw/new_variants.vcf --output results/predictions.csv
   ```

5. **View Results**
   ```bash
   # Open MLflow UI
   mlflow ui
   
   # Check results directory
   ls -la results/
   ```

##  Troubleshooting

### **Common Issues**

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **XGBoost Issues (macOS)**
   ```bash
   brew install libomp
   ```

3. **MLflow Connection Issues**
   ```bash
   export MLFLOW_TRACKING_URI="http://localhost:5000"
   ```

4. **Data Path Issues**
   - Ensure VCF files are in `data/raw/`
   - Check file permissions
   - Verify file format

### **Getting Help**
- Check logs in `logs/` directory
- Review MLflow UI for experiment details
- Run `python test_pipeline.py` for basic functionality test
- Check GitHub Issues for common problems

##  Success Indicators

 **Pipeline is working when you see:**
- Test script runs without errors
- MLflow UI shows experiments
- Models are saved in `models/` directory
- Results are generated in `results/` directory
- Logs show successful completion

