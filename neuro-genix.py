# genetic_analyzer_main.py
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqUtils
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
class AdvancedDiseasePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
        super(AdvancedDiseasePredictor, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=1)
        )

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer5 = nn.Linear(hidden_size // 4, num_classes)

        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size // 2)
        self.batchnorm4 = nn.BatchNorm1d(hidden_size // 4)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        attention_weights = self.attention(x)
        x = x * attention_weights

        identity = x

        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm3(self.layer3(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm4(self.layer4(x)))
        x = self.dropout(x)

        if identity.shape[1] == x.shape[1]:
            x = x + identity

        x = self.layer5(x)
        return x

class SmartGeneticAnalyzer:
    def __init__(self, model_paths=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_performance = {}
        self.best_model = None

        # Default paths
        default_paths = {
            'dl_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_dl_model.pth',
            'xgboost_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_xgboost_model.pkl',
            'random_forest_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_random_forest_model.pkl',
            'svm_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_svm_model.pkl',
            'scalers': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_scalers.pkl',
            'encoders': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_encoders.pkl'
        }

        # Update with custom paths if provided
        if model_paths:
            default_paths.update(model_paths)

        self.model_paths = default_paths
        self.load_all_models()
        self.analyze_model_performance()

    def load_all_models(self):
        """Load all trained models with individual paths"""
        print("Loading all trained models...")
        print("Model paths being used:")
        for model_type, path in self.model_paths.items():
            print(f"  {model_type}: {path}")

        try:
            # Load preprocessing objects
            if os.path.exists(self.model_paths['scalers']):
                self.scalers = joblib.load(self.model_paths['scalers'])
                print("✓ Scalers loaded")
            else:
                print("✗ Scalers file not found")
                return False

            if os.path.exists(self.model_paths['encoders']):
                self.label_encoders = joblib.load(self.model_paths['encoders'])
                print("✓ Label encoders loaded")
            else:
                print("✗ Encoders file not found")
                return False

            # Load deep learning model
            if os.path.exists(self.model_paths['dl_model']):
                input_size = len(self.scalers['features'].scale_)
                hidden_size = 256
                num_classes = len(self.label_encoders['disease'].classes_)

                dl_model = AdvancedDiseasePredictor(input_size, hidden_size, num_classes)
                dl_model.load_state_dict(torch.load(self.model_paths['dl_model'], map_location=self.device))
                dl_model.to(self.device)
                dl_model.eval()
                self.models['deep_learning'] = dl_model
                print("✓ Deep Learning model loaded")
            else:
                print("✗ Deep Learning model file not found")

            # Load other models
            model_files = {
                'xgboost': self.model_paths['xgboost_model'],
                'random_forest': self.model_paths['random_forest_model'],
                'svm': self.model_paths['svm_model']
            }

            for model_name, model_file in model_files.items():
                if os.path.exists(model_file):
                    try:
                        self.models[model_name] = joblib.load(model_file)
                        print(f"✓ {model_name.replace('_', ' ').title()} model loaded")
                    except Exception as e:
                        print(f"✗ Error loading {model_name}: {e}")
                else:
                    print(f"✗ {model_name} model file not found: {model_file}")

            if not self.models:
                print("No models were loaded successfully!")
                return False

            print("All available models loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def analyze_model_performance(self):
        """Analyze which model performs best"""
        print("Analyzing model performance...")

        self.model_performance = {}
        for model_name in self.models.keys():
            self.model_performance[model_name] = {
                'avg_confidence': 0.8,  # Placeholder
                'consistency': 0.9,     # Placeholder
            }

        # Simple best model selection (you can enhance this)
        if self.models:
            self.best_model = list(self.models.keys())[0]
            print(f"Best model selected: {self.best_model}")

    def _predict_single_model(self, model_name, model, sequence):
        """Predict using a single model"""
        features = self.extract_advanced_features(sequence)
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scalers['features'].transform(features_array)

        if model_name == 'deep_learning':
            with torch.no_grad():
                outputs = model(torch.FloatTensor(features_scaled).to(self.device))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]
                disease_name = self.label_encoders['disease'].inverse_transform([pred_class])[0]
        else:
            probs = model.predict_proba(features_scaled)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            disease_name = self.label_encoders['disease'].inverse_transform([pred_class])[0]

        return {
            'prediction': disease_name,
            'confidence': confidence,
            'probabilities': probs
        }

    def extract_advanced_features(self, sequence):
        """Extract comprehensive features from genetic sequence"""
        try:
            # Basic sequence features
            gc_content = SeqUtils.GC(Seq(sequence.upper()))
            at_content = 100 - gc_content
            seq_length = len(sequence)

            # Advanced features
            codon_bias = self.calculate_codon_bias(sequence)
            repeat_density = self.calculate_repeat_density(sequence)
            mutation_density = self.calculate_mutation_density(sequence)
            entropy = self.calculate_sequence_entropy(sequence)
            cg_content = self.calculate_cg_content(sequence)
            purine_content = self.calculate_purine_content(sequence)
            kmer_complexity = self.calculate_kmer_complexity(sequence)
            palindromic_density = self.calculate_palindromic_density(sequence)

            return [
                gc_content, at_content, seq_length,
                codon_bias, repeat_density, mutation_density,
                entropy, cg_content, purine_content,
                kmer_complexity, palindromic_density
            ]
        except:
            return [0] * 11

    def calculate_codon_bias(self, sequence):
        """Calculate codon usage bias"""
        if len(sequence) < 3:
            return 0
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        if not codons:
            return 0
        unique_codons = len(set(codons))
        return unique_codons / len(codons)

    def calculate_repeat_density(self, sequence):
        """Calculate repeat sequence density"""
        repeats = 0
        for i in range(len(sequence)-3):
            if sequence[i] == sequence[i+1] == sequence[i+2]:
                repeats += 1
        return repeats / max(len(sequence), 1) * 1000

    def calculate_mutation_density(self, sequence):
        """Calculate mutation density"""
        mutations = 0
        sequence = sequence.upper()
        for i in range(len(sequence)-1):
            if (sequence[i] in 'CG' and sequence[i+1] in 'CG'):
                mutations += 1
        return mutations / max(len(sequence), 1) * 1000

    def calculate_sequence_entropy(self, sequence):
        """Calculate sequence entropy"""
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p + 1e-10)
        return entropy

    def calculate_cg_content(self, sequence):
        """Calculate CG content"""
        sequence = sequence.upper()
        cg_count = sequence.count('C') + sequence.count('G')
        return (cg_count / len(sequence)) * 100 if sequence else 0

    def calculate_purine_content(self, sequence):
        """Calculate purine content"""
        sequence = sequence.upper()
        purine_count = sequence.count('A') + sequence.count('G')
        return (purine_count / len(sequence)) * 100 if sequence else 0

    def calculate_kmer_complexity(self, sequence, k=3):
        """Calculate k-mer complexity"""
        if len(sequence) < k:
            return 0
        kmers = set()
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i:i+k])
        return len(kmers) / (len(sequence) - k + 1)

    def calculate_palindromic_density(self, sequence):
        """Calculate palindromic density"""
        palindromes = 0
        for i in range(len(sequence) - 5):
            substr = sequence[i:i+6]
            if substr == substr[::-1]:
                palindromes += 1
        return palindromes / max(len(sequence) - 5, 1) * 100

    def select_best_model_for_sequence(self, sequence):
        """Select the best model based on sequence characteristics"""
        # Simple selection logic - use the first available model
        if self.models:
            return list(self.models.keys())[0]
        return None

    def predict_disease(self, genetic_sequence):
        """Smart prediction using the best model"""
        if not genetic_sequence or len(genetic_sequence.strip()) < 20:
            return {'error': 'Sequence too short. Minimum 20 characters required.'}

        valid_chars = set('ATGCatgc')
        if not all(char in valid_chars for char in genetic_sequence.upper()):
            return {'error': 'Invalid characters. Only A, T, G, C allowed.'}

        try:
            best_model_name = self.select_best_model_for_sequence(genetic_sequence)
            if not best_model_name:
                return {'error': 'No models available for prediction'}

            best_model = self.models[best_model_name]
            best_result = self._predict_single_model(best_model_name, best_model, genetic_sequence)

            # Get predictions from all available models
            all_predictions = {}
            all_confidences = {}
            all_probs = {}

            for model_name, model in self.models.items():
                try:
                    result = self._predict_single_model(model_name, model, genetic_sequence)
                    all_predictions[model_name] = result['prediction']
                    all_confidences[model_name] = result['confidence']
                    all_probs[model_name] = result['probabilities']
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    continue

            # Calculate consensus
            predictions_list = list(all_predictions.values())
            consensus = max(set(predictions_list), key=predictions_list.count) if predictions_list else best_result['prediction']

            # Average probabilities
            disease_classes = self.label_encoders['disease'].classes_
            avg_probs = np.mean([all_probs[model] for model in all_probs], axis=0)
            disease_probabilities = {disease: prob for disease, prob in zip(disease_classes, avg_probs)}

            return {
                'best_model_used': best_model_name,
                'consensus_prediction': consensus,
                'best_model_prediction': best_result['prediction'],
                'best_model_confidence': best_result['confidence'],
                'all_predictions': all_predictions,
                'all_confidences': all_confidences,
                'disease_probabilities': disease_probabilities,
                'extracted_features': dict(zip([
                    'GC Content (%)', 'AT Content (%)', 'Sequence Length',
                    'Codon Bias', 'Repeat Density', 'Mutation Density',
                    'Entropy', 'CG Content', 'Purine Content',
                    'K-mer Complexity', 'Palindromic Density'
                ], self.extract_advanced_features(genetic_sequence))),
                'sequence_length': len(genetic_sequence)
            }

        except Exception as e:
            return {'error': f'Analysis error: {str(e)}'}

# Gradio Interface Functions
def analyze_sequence(sequence):
    """Main analysis function for Gradio"""
    if not sequence:
        return "Please enter a genetic sequence", None, None, None, None

    result = analyzer.predict_disease(sequence)

    if 'error' in result:
        return result['error'], None, None, None, None

    # Create results text
    results_text = f"""## Smart Genetic Sequence Analysis

** Best Model Used:** **{result['best_model_used'].replace('_', ' ').title()}** (Confidence: {result['best_model_confidence']:.1%})

** Consensus Prediction:** **{result['consensus_prediction']}**

** Sequence Length:** {result['sequence_length']} bases

###  Model Performance:
"""

    for model, prediction in result['all_predictions'].items():
        confidence = result['all_confidences'][model]
        model_display = model.replace('_', ' ').title()
        results_text += f"- **{model_display}:** {prediction} (Confidence: {confidence:.1%})\n"
        if model == result['best_model_used']:
            results_text += "  ⭐ *Selected as best for this sequence*\n"

    results_text += "\n### 🔍 Sequence Characteristics:\n"
    features = result['extracted_features']
    for feature, value in features.items():
        results_text += f"- **{feature}:** {value:.4f}\n"

    # Create visualizations
    fig1 = create_feature_plot(features)
    fig2 = create_probability_plot(result['disease_probabilities'])
    fig3 = create_confidence_plot(result['all_confidences'])
    fig4 = create_model_comparison_plot(result['all_confidences'], result['best_model_used'])

    return results_text, fig1, fig2, fig3, fig4

def create_feature_plot(features):
    fig, ax = plt.subplots(figsize=(12, 6))
    features_names = list(features.keys())
    values = list(features.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(features_names)))
    bars = ax.bar(features_names, values, color=colors)

    ax.set_title('Extracted Genetic Sequence Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def create_probability_plot(probabilities):
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    diseases = list(sorted_probs.keys())
    probs = list(sorted_probs.values())

    colors = plt.cm.plasma(np.linspace(0, 1, len(diseases)))
    bars = ax.bar(range(len(diseases)), probs, color=colors)

    ax.set_title('Disease Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(range(len(diseases)))
    ax.set_xticklabels(diseases, rotation=45, ha='right')
    ax.set_ylim(0, max(probs) * 1.1)

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        if prob > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def create_confidence_plot(confidence_scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [m.replace('_', ' ').title() for m in confidence_scores.keys()]
    confidences = list(confidence_scores.values())

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, confidences, color=colors)

    ax.set_title('Model Confidence Scores', fontsize=14, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_ylim(0, 1.1)

    for bar, confidence in zip(bars, confidences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{confidence:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_model_comparison_plot(confidence_scores, best_model):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [m.replace('_', ' ').title() for m in confidence_scores.keys()]
    confidences = list(confidence_scores.values())

    colors = ['lightblue' if m != best_model.replace('_', ' ').title() else 'gold' for m in models]

    bars = ax.bar(models, confidences, color=colors, edgecolor='black', linewidth=1)

    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_ylim(0, 1.1)

    for bar, confidence, model in zip(bars, confidences, models):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{confidence:.1%}', ha='center', va='bottom', fontweight='bold')

        if model == best_model.replace('_', ' ').title():
            ax.text(bar.get_x() + bar.get_width()/2., -0.1,
                    ' BEST', ha='center', va='top', fontweight='bold', color='darkorange')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def show_example_sequence():
    return "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

# Global analyzer instance
analyzer = None

def main():
    global analyzer

    # For Colab/Jupyter environments, use default paths without argparse
    model_paths = {
        'dl_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_dl_model.pth',
        'xgboost_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_xgboost_model.pkl',
        'random_forest_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_random_forest_model.pkl',
        'svm_model': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_svm_model.pkl',
        'scalers': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_scalers.pkl',
        'encoders': '/content/drive/MyDrive/Neuro Genix/models/genetic_analyzer_models_encoders.pkl'
    }

    # Initialize the analyzer with default paths
    print("Loading models with default paths:")
    for model_type, path in model_paths.items():
        print(f"  {model_type}: {path}")

    analyzer = SmartGeneticAnalyzer(model_paths=model_paths)

    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Smart Genetic Sequence Analyzer") as demo:
        gr.Markdown("""
        # Smart Genetic Sequence Analyzer
        ##  AI-Powered Disease Prediction with Model Selection
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("###  Input Genetic Sequence")
                sequence_input = gr.Textbox(
                    label="DNA Sequence (ATCG format)",
                    placeholder="Enter genetic sequence (e.g., ATCGATCGATCG...)",
                    lines=5,
                    max_lines=10,
                    value=show_example_sequence()
                )

                with gr.Row():
                    example_btn = gr.Button(" Load Example")
                    analyze_btn = gr.Button(" Analyze Sequence", variant="primary")

                gr.Markdown("""
                ###  Instructions:
                - Enter a valid DNA sequence (A, T, C, G characters only)
                - Minimum length: 20 characters
                - The system will automatically select the best AI model
                """)

            with gr.Column(scale=2):
                gr.Markdown("###  Analysis Results")
                results_output = gr.Markdown(label="Analysis Summary")

                with gr.Row():
                    feature_plot = gr.Plot(label="Sequence Features")
                    prob_plot = gr.Plot(label="Disease Probabilities")

                with gr.Row():
                    confidence_plot = gr.Plot(label="Model Confidence Scores")
                    model_plot = gr.Plot(label="Model Comparison")

        example_btn.click(fn=show_example_sequence, outputs=sequence_input)
        analyze_btn.click(fn=analyze_sequence, inputs=sequence_input,
                         outputs=[results_output, feature_plot, prob_plot, confidence_plot, model_plot])

        gr.Markdown("""
        ---
        **  This tool is for research purposes only.
        """)

    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()