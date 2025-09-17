# Gene Variant Analysis Pipeline

A comprehensive MLOps-enabled pipeline for analyzing genomic variants using machine learning techniques.

## Features

- **Advanced Variant Analysis**: Comprehensive feature engineering for genomic variants
- **Multiple ML Models**: Support for XGBoost, LightGBM, Random Forest, and ensemble methods
- **MLOps Integration**: MLflow tracking, DVC version control, and automated CI/CD
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Scalable Architecture**: Modular design for easy extension and customization

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rodolfopadilla/gene-variant-analysis.git
cd gene-variant-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

### Usage

#### Training a Model

```bash
python scripts/train.py --vcf data/raw/variants.vcf --model-type xgboost
```

#### Using Docker

```bash
docker-compose up --build
```

#### Available Scripts

- `gva-train`: Train models on variant data
- `gva-evaluate`: Evaluate model performance
- `gva-predict`: Make predictions on new data

## Project Structure

```
Gene-Variant-Analysis/
├── .github/workflows/          # CI/CD workflows
├── config/                     # Configuration files
├── data/                       # Data directories
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   ├── external/              # External data
│   └── interim/               # Intermediate data
├── models/                     # Model artifacts
│   ├── trained/               # Trained models
│   └── artifacts/             # Model artifacts
├── src/                        # Source code
│   ├── data/                  # Data processing
│   ├── features/              # Feature engineering
│   ├── models/                # ML models
│   ├── pipeline/              # Training pipelines
│   ├── utils/                 # Utilities
│   └── visualization/         # Visualization
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Test suite
├── scripts/                    # Command-line scripts
├── docker/                     # Docker configuration
└── docs/                       # Documentation
```

## Configuration

The pipeline is configured via `config/config.yaml`. Key configuration options:

- **Data paths**: Raw, processed, and external data directories
- **Model parameters**: Hyperparameters for different ML models
- **Feature engineering**: Feature extraction settings
- **MLflow**: Experiment tracking configuration

## Development

### Setting up Development Environment

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest
```

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

## MLOps Features

### MLflow Integration

- Experiment tracking
- Model versioning
- Artifact storage
- Model registry

### DVC Integration

- Data versioning
- Pipeline reproducibility
- Data lineage tracking

### CI/CD Pipeline

- Automated testing
- Code quality checks
- Model deployment
- Documentation generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gene_variant_analysis,
  title={Gene Variant Analysis Pipeline},
  author={Rodolfo Padilla},
  year={2024},
  url={https://github.com/rodolfopadilla/gene-variant-analysis}
}
```

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.

## Changelog

### v1.0.0
- Initial release
- Core pipeline functionality
- MLOps integration
- Docker support
