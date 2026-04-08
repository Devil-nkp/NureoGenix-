# 🧬 NureoGenix – AI-Powered Genetic Disease Predictor

NureoGenix is an intelligent bioinformatics platform designed to revolutionize genetic analysis by predicting disease predisposition from DNA sequences using advanced deep learning models, ensemble machine learning, and comprehensive genetic feature extraction with interactive Gradio interface.

---

## Overview

This system combines PyTorch neural networks with ensemble methods (XGBoost, Random Forest, SVM) to deliver accurate disease prediction from genetic sequences through intelligent feature extraction and multi-model consensus prediction.

Instead of relying solely on single prediction models, NureoGenix:

1. Extracts advanced genetic features from DNA sequences (11+ dimensional features)
2. Uses attention-based deep learning for improved accuracy
3. Implements ensemble voting from multiple AI models
4. Provides confidence scores and probability distributions
5. Analyzes sequence characteristics (GC content, entropy, codon bias, etc.)
6. Generates interactive visualizations with Gradio interface
7. Delivers consensus predictions across all trained models

---

## Architecture

DNA Sequence Input (ATCG format)
→ Advanced Feature Extraction (11 Genetic Features)
→ Feature Scaling & Normalization
→ Multi-Model Processing:
  - Deep Learning (Attention + Batch Norm)
  - XGBoost (Gradient Boosting)
  - Random Forest (Ensemble)
  - SVM (Support Vector Machine)
→ Consensus Prediction & Voting
→ Visualization & Results
→ Interactive Gradio Dashboard

---

## ⚙️ Tech Stack

- **HTML** (66.3%)
- **Python** (33.7%)
- **PyTorch** (Deep Learning Framework)
- **TensorFlow / Keras** (Neural Networks)
- **Scikit-learn** (Machine Learning)
- **XGBoost** (Gradient Boosting)
- **Biopython** (Bioinformatics)
- **Gradio** (Interactive UI)
- **Pandas** (Data Processing)
- **NumPy** (Numerical Computing)
- **Matplotlib & Seaborn** (Visualization)
- **Joblib** (Model Serialization)

---

## Key Features

- **Attention-Based Deep Learning** – Advanced neural network with attention mechanism and batch normalization
- **Multi-Model Ensemble** – Combines Deep Learning, XGBoost, Random Forest, and SVM
- **Advanced Feature Extraction** – 11 genetic features including GC content, entropy, codon bias, repeat density, k-mer complexity
- **Consensus Prediction** – Voting mechanism across all models for robust predictions
- **Sequence Validation** – Automatic validation for valid DNA sequences (A, T, G, C only)
- **Confidence Scoring** – Individual model confidence scores and probabilities
- **Disease Probability Distribution** – Detailed breakdown of disease probabilities
- **Interactive Visualizations** – 4 interactive plots (features, probabilities, confidence, comparison)
- **Model Performance Analysis** – Accuracy, Precision, Recall, F1-Score for all models
- **Early Stopping** – Prevents overfitting during deep learning training
- **Class Weight Balancing** – Handles imbalanced disease data
- **Gradio Web Interface** – User-friendly interactive dashboard

---

## Live Demo

🔗 Coming Soon

---

## Installation (Local Setup)

```bash
git clone https://github.com/Devil-nkp/NureoGenix-.git
cd NureoGenix-
pip install -r requirements.txt
```

Set up your environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install xgboost scikit-learn pandas numpy
pip install biopython matplotlib seaborn
pip install gradio joblib
```

Download trained models or train new ones:

```bash
# Place your dataset at: ./Data/newdata.csv
# Dataset should have columns: 'genetic_sequence', 'disease'

# Train models (optional)
python "model creation.py"

# Run the interactive interface
python "neuro-genix.py"
```

Open your browser at `http://localhost:7860`

---

## Project Structure

```
NureoGenix-/
├── neuro-genix.py              # Main interactive Gradio application
├── model creation.py           # Model training script
├── index.html                  # Web interface (66.3% HTML)
├── Data/                       # Dataset directory
│   └── newdata.csv            # Training dataset with sequences and diseases
└── README.md                   # Project documentation
```

---

## Usage Example

```python
from neuro_genix import SmartGeneticAnalyzer

# Initialize analyzer with trained models
analyzer = SmartGeneticAnalyzer(model_paths={
    'dl_model': 'models/genetic_analyzer_models_dl_model.pth',
    'xgboost_model': 'models/genetic_analyzer_models_xgboost_model.pkl',
    'random_forest_model': 'models/genetic_analyzer_models_random_forest_model.pkl',
    'svm_model': 'models/genetic_analyzer_models_svm_model.pkl',
    'scalers': 'models/genetic_analyzer_models_scalers.pkl',
    'encoders': 'models/genetic_analyzer_models_encoders.pkl'
})

# Predict disease from DNA sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
results = analyzer.predict_disease(sequence)

print(f"Consensus Prediction: {results['consensus_prediction']}")
print(f"Confidence: {results['best_model_confidence']:.1%}")
print(f"Disease Probabilities: {results['disease_probabilities']}")
```

---

## Genetic Features Extracted

1. **GC Content** (%) - Percentage of G and C bases
2. **AT Content** (%) - Percentage of A and T bases
3. **Sequence Length** - Total number of bases
4. **Codon Bias** - Codon usage variation
5. **Repeat Density** - Frequency of repeated sequences
6. **Mutation Density** - CpG islands and mutation hotspots
7. **Entropy** - Shannon entropy of sequence
8. **CG Content** - Specific C and G content
9. **Purine Content** - A and G content percentage
10. **K-mer Complexity** - Diversity of 3-mers
11. **Palindromic Density** - Frequency of palindromic sequences

---

## Model Performance

- **Deep Learning Model**: Attention-based, 5-layer neural network with batch normalization
- **XGBoost Model**: 1000 trees with early stopping, optimized hyperparameters
- **Random Forest Model**: 100 trees with max depth 10
- **SVM Model**: RBF kernel with probability estimates

Expected Accuracy: 85-95% (varies with dataset)

---

## Future Improvements

- Add pathway analysis and gene ontology integration
- Implement SHAP values for model interpretability
- Add support for RNA sequences and protein predictions
- Develop RESTful API for integration
- Implement real-time model retraining
- Add database integration for variant storage
- Create mobile app for sequence analysis
- Implement multi-threading for batch processing
- Add support for VCF file format
- Integrate with external bioinformatics databases

---

## Author

**Naveenkumar G** (Devil-nkp)
- AI / ML Engineer
- Bioinformatics Developer

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ to advance personalized genomic medicine and disease prediction**
