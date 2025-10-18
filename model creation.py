# genetic_analyzer_complete.py (CORRECTED VERSION)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from Bio import SeqIO, SeqUtils
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Advanced Dataset Class
class GeneticDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Advanced Deep Learning Model
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
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Deep network with skip connections
        identity = x

        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm3(self.layer3(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm4(self.layer4(x)))
        x = self.dropout(x)

        # Skip connection
        if identity.shape[1] == x.shape[1]:
            x = x + identity

        x = self.layer5(x)
        return x

# Main Genetic Analyzer Class
class AdvancedGeneticAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.is_trained = False

    def load_and_preprocess_data(self):
        """Load and preprocess genetic dataset with advanced features"""
        print("Loading and preprocessing data...")

        try:
            # Load dataset
            df = pd.read_csv(self.dataset_path)

            # Validate dataset structure
            required_columns = ['genetic_sequence', 'disease']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Dataset must contain 'genetic_sequence' and 'disease' columns")

            # Extract advanced features
            print("Extracting advanced sequence features...")
            df['sequence_features'] = df['genetic_sequence'].apply(self.extract_advanced_features)

            # Enhanced feature extraction
            feature_columns = [
                'gc_content', 'at_content', 'sequence_length',
                'codon_bias', 'repeat_density', 'mutation_density',
                'entropy', 'cg_content', 'purine_content',
                'kmer_complexity', 'palindromic_density'
            ]

            for i, col in enumerate(feature_columns):
                df[col] = df['sequence_features'].apply(lambda x: x[i])

            # Encode target diseases
            self.label_encoders['disease'] = LabelEncoder()
            df['disease_encoded'] = self.label_encoders['disease'].fit_transform(df['disease'])

            # Prepare features and target
            X = df[feature_columns]
            y = df['disease_encoded']

            # Handle class imbalance with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42,
                stratify=y, shuffle=True
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42,
                stratify=y_train
            )

            # Advanced scaling with robust scaler
            self.scalers['features'] = StandardScaler()
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_val_scaled = self.scalers['features'].transform(X_val)
            X_test_scaled = self.scalers['features'].transform(X_test)

            print(f"Dataset loaded: {len(df)} sequences, {len(df['disease'].unique())} diseases")
            print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

            return (X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test, feature_columns, df)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def extract_advanced_features(self, sequence):
        """Extract comprehensive advanced features from genetic sequence"""
        try:
            seq_obj = Seq(sequence.upper())

            # Basic sequence features
            gc_content = SeqUtils.GC(seq_obj)
            at_content = 100 - gc_content
            seq_length = len(sequence)

            # Advanced features
            codon_bias = self.calculate_codon_bias(sequence)
            repeat_density = self.calculate_repeat_density(sequence)
            mutation_density = self.calculate_mutation_density(sequence)

            # New advanced features
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
        """Calculate advanced codon usage bias"""
        if len(sequence) < 3:
            return 0

        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        if not codons:
            return 0

        codon_counts = {}
        for codon in codons:
            codon_counts[codon] = codon_counts.get(codon, 0) + 1

        # Calculate Shannon entropy of codon usage
        total = len(codons)
        entropy = 0
        for count in codon_counts.values():
            p = count / total
            entropy -= p * np.log2(p + 1e-10)

        return entropy / np.log2(len(codon_counts) + 1e-10)

    def calculate_repeat_density(self, sequence):
        """Calculate repeat sequence density with multiple patterns"""
        repeats = 0
        # Check for various repeat patterns
        for pattern_length in [2, 3, 4]:
            for i in range(len(sequence) - pattern_length):
                pattern = sequence[i:i+pattern_length]
                if sequence.count(pattern) > 2:  # More than 2 occurrences
                    repeats += 1
        return repeats / max(len(sequence), 1) * 100

    def calculate_mutation_density(self, sequence):
        """Calculate potential mutation density with advanced patterns"""
        mutations = 0
        sequence = sequence.upper()

        # CpG islands
        for i in range(len(sequence)-1):
            if sequence[i] == 'C' and sequence[i+1] == 'G':
                mutations += 2  # Higher weight for CpG

        # TATA boxes and other motifs
        motifs = ['TATA', 'ATAT', 'CGCG', 'GCGC']
        for motif in motifs:
            mutations += sequence.count(motif)

        return mutations / max(len(sequence), 1) * 100

    def calculate_sequence_entropy(self, sequence):
        """Calculate Shannon entropy of the sequence"""
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p + 1e-10)
        return entropy

    def calculate_cg_content(self, sequence):
        """Calculate specific CG content"""
        sequence = sequence.upper()
        cg_count = sequence.count('C') + sequence.count('G')
        return (cg_count / len(sequence)) * 100 if sequence else 0

    def calculate_purine_content(self, sequence):
        """Calculate purine (A/G) content"""
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
        """Calculate palindromic sequence density"""
        palindromes = 0
        for i in range(len(sequence) - 5):  # Check for 6-base palindromes
            substr = sequence[i:i+6]
            if substr == substr[::-1]:
                palindromes += 1
        return palindromes / max(len(sequence) - 5, 1) * 100

    def train_deep_learning_model(self, X_train, y_train, X_val, y_val, num_classes):
        """Train advanced deep learning model with attention mechanism"""
        print("Training Advanced Deep Learning Model...")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val.values).to(self.device)

        # Create advanced model
        input_size = X_train.shape[1]
        hidden_size = 256
        model = AdvancedDiseasePredictor(input_size, hidden_size, num_classes).to(self.device)

        # Advanced training setup
        criterion = nn.CrossEntropyLoss(weight=self.calculate_class_weights(y_train))
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop with early stopping
        num_epochs = 200
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_acc = (torch.argmax(val_outputs, dim=1) == y_val_tensor).float().mean()

            scheduler.step(val_loss)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc.item())

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_dl_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}')

        # Load best model
        model.load_state_dict(torch.load('best_dl_model.pth'))
        self.models['deep_learning'] = model

        return train_losses, val_losses, val_accuracies

    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return torch.FloatTensor(weights).to(self.device)

    def train_xgboost_model(self, X_train, y_train, X_val, y_val):
        """Train advanced XGBoost model with hyperparameter optimization"""
        print("Training Advanced XGBoost Model...")

        # Advanced XGBoost with optimized parameters
        model = xgb.XGBClassifier(
            n_estimators=1000,  # Reduced for faster training
            learning_rate=0.01,
            max_depth=6,  # Reduced depth
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            gamma=0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method='hist' if torch.cuda.is_available() else 'auto',
            n_jobs=-1  # Use all available cores
        )

        # FIXED: Use proper parameter names for current XGBoost version
        try:
            # Try with early stopping (newer versions)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
        except TypeError:
            # Fallback for older versions
            print("Using fallback XGBoost training (no early stopping)")
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100
            )

        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = model.feature_importances_

        return model

    def train_ensemble_model(self, X_train, y_train):
        """Train advanced ensemble model"""
        print("Training Ensemble Model...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        # Use simpler ensemble approach
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        svc_model = SVC(
            probability=True,
            random_state=42,
            kernel='rbf',
            C=1.0
        )

        # Train individual models
        print("Training Random Forest...")
        rf_model.fit(X_train, y_train)

        print("Training SVM...")
        svc_model.fit(X_train, y_train)

        # Store individual models
        self.models['random_forest'] = rf_model
        self.models['svm'] = svc_model

        return rf_model, svc_model

    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\nEvaluating Models...")
        results = {}

        for model_name, model in self.models.items():
            if model_name == 'deep_learning':
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.FloatTensor(X_test).to(self.device))
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            accuracy = np.mean(predictions == y_test.values)

            # Additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': predictions,
                'probabilities': probabilities
            }

            print(f"\n{model_name.upper()} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

        return results

    def train_all_models(self):
        """Main training function"""
        print("Starting Advanced Genetic Analyzer Training...")

        # Load and preprocess data
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         feature_columns, df) = self.load_and_preprocess_data()

        num_classes = len(df['disease'].unique())

        # Train all models
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)

        # Train Deep Learning model
        print("\n1. Training Deep Learning Model...")
        self.train_deep_learning_model(X_train, y_train, X_val, y_val, num_classes)

        # Train XGBoost model
        print("\n2. Training XGBoost Model...")
        self.train_xgboost_model(X_train, y_train, X_val, y_val)

        # Train Ensemble models
        print("\n3. Training Ensemble Models...")
        self.train_ensemble_model(X_train, y_train)

        # Evaluate models
        results = self.evaluate_models(X_test, y_test)

        # Save models
        self.save_models("genetic_analyzer_models")

        self.is_trained = True
        print("\nTraining completed successfully!")

        return results

    def save_models(self, filepath):
        """Save all trained models and preprocessing objects"""
        # Save deep learning model
        torch.save(self.models['deep_learning'].state_dict(), f'{filepath}_dl_model.pth')

        # Save other models
        for model_name, model in self.models.items():
            if model_name != 'deep_learning':
                joblib.dump(model, f'{filepath}_{model_name}_model.pkl')

        # Save preprocessing objects
        joblib.dump(self.scalers, f'{filepath}_scalers.pkl')
        joblib.dump(self.label_encoders, f'{filepath}_encoders.pkl')

        print(f"Models saved to {filepath}_*.pth/pkl")

    def load_models(self, filepath="genetic_analyzer_models"):
        """Load trained models"""
        try:
            # Load preprocessing objects
            self.scalers = joblib.load(f'{filepath}_scalers.pkl')
            self.label_encoders = joblib.load(f'{filepath}_encoders.pkl')

            # Load deep learning model
            input_size = len(self.scalers['features'].scale_)
            hidden_size = 256
            num_classes = len(self.label_encoders['disease'].classes_)

            model = AdvancedDiseasePredictor(input_size, hidden_size, num_classes)
            model.load_state_dict(torch.load(f'{filepath}_dl_model.pth', map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models['deep_learning'] = model

            # Load other models
            model_types = ['xgboost', 'random_forest', 'svm']
            for model_type in model_types:
                try:
                    self.models[model_type] = joblib.load(f'{filepath}_{model_type}_model.pkl')
                except:
                    print(f"Model {model_type} not found, skipping...")

            self.is_trained = True
            print("Models loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Main execution
if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = "/content/drive/MyDrive/Neuro Genix/newdata.csv"

    # Create analyzer instance
    analyzer = AdvancedGeneticAnalyzer(dataset_path)

    # Train models (run this once)
    print("Training models from dataset...")
    results = analyzer.train_all_models()

    print("\nTraining completed! Models are ready for prediction.")