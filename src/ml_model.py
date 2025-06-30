import xgboost as xgb
from xgboost import XGBClassifier   
import numpy as np
import pandas as pd
import optuna
import json
import os
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, 
    precision_recall_curve
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import resample
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class XGBoostMultiClassClassifier:
    """XGBoost-based multi-class classifier for variant pathogenicity prediction."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.class_labels = ['Benign', 'Likely_benign', 'VUS', 'Likely_pathogenic', 'Pathogenic']
        self.label_encoder = LabelEncoder().fit(self.class_labels)
        self.categorical_encoders = {}
        
        self.params = params or {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': len(self.class_labels),
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.feature_names = None
        self.best_params = None

    def preprocess_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess features for XGBoost training/prediction."""
        X_processed = X.copy()
        
        # Handle categorical columns
        categorical_cols = ['chromosome', 'gene']
        
        for col in categorical_cols:
            if col in X_processed.columns:
                if is_training:
                    # Fit encoder during training
                    encoder = LabelEncoder()
                    X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
                    self.categorical_encoders[col] = encoder
                else:
                    # Use existing encoder during prediction
                    if col in self.categorical_encoders:
                        # Handle unseen categories
                        unique_vals = set(X_processed[col].astype(str))
                        known_vals = set(self.categorical_encoders[col].classes_)
                        unknown_vals = unique_vals - known_vals
                        
                        if unknown_vals:
                            # Replace unknown values with the most common known value
                            most_common = self.categorical_encoders[col].classes_[0]
                            X_processed[col] = X_processed[col].astype(str).replace(
                                list(unknown_vals), most_common
                            )
                        
                        X_processed[col] = self.categorical_encoders[col].transform(X_processed[col].astype(str))

        # Handle numeric columns and missing values
        numeric_defaults = {
            'sift_score': 0.5,
            'polyphen2_hdiv_score': 0.5,
            'cadd_phred': 0.0,
            'gerp_rs': 0.0,
            'freq_gnomad_genome_all': 0.0
        }
        
        for col, default in numeric_defaults.items():
            if col in X_processed.columns:
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(default)

        # Ensure all remaining columns are numeric
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)

        if is_training:
            self.feature_names = X_processed.columns.tolist()
        
        return X_processed

    def _apply_class_balancing(self, X_train: pd.DataFrame, y_train_encoded: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply class balancing using undersampling + SMOTE."""
        class_indices = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        
        # Undersample Benign class
        benign_mask = (y_train_encoded == class_indices['Benign'])
        other_mask = ~benign_mask

        X_benign = X_train[benign_mask]
        y_benign = y_train_encoded[benign_mask]

        # Limit benign samples to reasonable size
        benign_sample_size = min(500000, len(X_benign))
        if len(X_benign) > benign_sample_size:
            X_benign_down, y_benign_down = resample(
                X_benign, y_benign,
                replace=False,
                n_samples=benign_sample_size,
                random_state=42
            )
        else:
            X_benign_down, y_benign_down = X_benign, y_benign

        # Combine with other classes
        X_others = X_train[other_mask]
        y_others = y_train_encoded[other_mask]

        X_combined = pd.concat([X_benign_down, X_others])
        y_combined = np.concatenate([y_benign_down, y_others])

        print("Before SMOTE class distribution:", Counter(y_combined))

        # Apply SMOTE to minority classes
        smote_strategy = {
            class_indices['Likely_benign']: min(150000, benign_sample_size),
            class_indices['Likely_pathogenic']: min(100000, benign_sample_size),
            class_indices['Pathogenic']: min(100000, benign_sample_size)
        }

        smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)
        print("After SMOTE class distribution:", Counter(y_resampled))

        return X_resampled, y_resampled

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        print("Starting Optuna hyperparameter tuning...")

        X_processed = self.preprocess_features(X, is_training=True)
        y_encoded = self.label_encoder.transform(y)

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': len(self.class_labels),
                'random_state': 42,
                'n_jobs': -1,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5)
            }

            model = XGBClassifier(**params, use_label_encoder=False)
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
            score = cross_val_score(model, X_processed, y_encoded, scoring='f1_macro', cv=cv).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {study.best_value:.4f}")

        # Save best params
        with open("best_xgb_params.json", "w") as f:
            json.dump(self.best_params, f, indent=4)

        self.params.update(self.best_params)
        return self.best_params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, 
              tune_hyperparams: bool = False):
        """Train the XGBoost model."""
        print(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")

        # Preprocess features
        X_train_processed = self.preprocess_features(X_train, is_training=True)
        y_train_encoded = self.label_encoder.transform(y_train)

        print("Original class distribution:", Counter(y_train_encoded))

        # Apply class balancing
        X_resampled, y_resampled = self._apply_class_balancing(X_train_processed, y_train_encoded)

        # Tune hyperparameters if requested
        if tune_hyperparams:
            if os.path.exists("best_xgb_params.json"):
                with open("best_xgb_params.json", "r") as f:
                    self.params.update(json.load(f))
                print("Loaded saved hyperparameters.")
            else:
                print("Tuning hyperparameters...")
                # Use a sample for hyperparameter tuning if dataset is large
                sample_size = min(100000, len(X_train))
                X_sample, _, y_sample, _ = train_test_split(
                    X_train, y_train,
                    train_size=sample_size,
                    stratify=y_train,
                    random_state=42
                )
                self.tune_hyperparameters(X_sample, y_sample)

        # Initialize and train model
        self.model = XGBClassifier(**self.params, use_label_encoder=False)

        # Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_processed = self.preprocess_features(X_val, is_training=False)
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val_processed, y_val_encoded)]

        # Train model
        self.model.fit(
            X_resampled, y_resampled,
            eval_set=eval_set,
            verbose=False
        )

        print("✅ Model training complete.")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        X_processed = self.preprocess_features(X, is_training=False)
        probabilities = self.model.predict_proba(X_processed)
        predictions = self.model.predict(X_processed)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities

    def _plot_roc_curves(self, y_true_bin: np.ndarray, y_proba: np.ndarray):
        """Plot ROC curves for each class."""
        fpr, tpr, roc_auc = {}, {}, {}
        
        plt.figure(figsize=(10, 8))
        for i in range(len(self.class_labels)):
            if np.any(y_true_bin[:, i]):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], label=f'{self.class_labels[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_precision_recall_curves(self, y_true_bin: np.ndarray, y_proba: np.ndarray):
        """Plot Precision-Recall curves for each class."""
        precision, recall, pr_auc = {}, {}, {}
        
        plt.figure(figsize=(10, 8))
        for i in range(len(self.class_labels)):
            if np.any(y_true_bin[:, i]):
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                pr_auc[i] = auc(recall[i], precision[i])
                plt.plot(recall[i], precision[i], label=f'{self.class_labels[i]} (AUC = {pr_auc[i]:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves (One-vs-Rest)')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig('pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        # Make predictions
        y_pred, y_proba = self.predict(X)
        
        print("\n📊 Evaluation Results:")
        print("=" * 50)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_labels, zero_division=0))
        
        # Prediction distribution
        print("\nPrediction Distribution:")
        pred_counts = Counter(y_pred)
        for cls in self.class_labels:
            count = pred_counts.get(cls, 0)
            percentage = (count / len(y_pred)) * 100
            print(f"{cls}: {count} ({percentage:.1f}%)")
        
        # ROC-AUC and PR-AUC
        try:
            y_true_encoded = self.label_encoder.transform(y_true)
            y_true_bin = label_binarize(y_true_encoded, classes=range(len(self.class_labels)))
            
            if y_proba.shape[1] == y_true_bin.shape[1]:
                metrics['auc_roc_macro'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                metrics['auc_pr_macro'] = average_precision_score(y_true_bin, y_proba, average='macro')
                
                # Generate plots
                self._plot_roc_curves(y_true_bin, y_proba)
                self._plot_precision_recall_curves(y_true_bin, y_proba)
                
            else:
                metrics['auc_roc_macro'] = 0.0
                metrics['auc_pr_macro'] = 0.0
        except Exception as e:
            print(f"Error calculating AUC metrics: {e}")
            metrics['auc_roc_macro'] = 0.0
            metrics['auc_pr_macro'] = 0.0
        
        # Print summary metrics
        print("\nSummary Metrics:")
        for key, value in metrics.items():
            print(f"{key.upper()}: {value:.4f}")
        
        return metrics

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Train the model first.")
            
        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        print(f"\nTop {top_n} Feature Importances:")
        print("=" * 40)
        for i, (feat, val) in enumerate(list(sorted_importance.items())[:top_n]):
            print(f"{i+1:2d}. {feat}: {val:.6f}")
            
        return sorted_importance
    
    def save_model(self, path: str):
        """Save the trained model and metadata."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Save model
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

        # Save metadata
        metadata = {
            "class_labels": self.class_labels,
            "params": self.params,
            "feature_names": self.feature_names,
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "categorical_encoders": {
                k: v.classes_.tolist() for k, v in self.categorical_encoders.items()
            }
        }
        
        metadata_path = str(path).replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model and metadata."""
        # Load model
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")

        # Load metadata
        metadata_path = model_path.replace(".pkl", "_metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.class_labels = metadata["class_labels"]
            self.params = metadata["params"]
            self.feature_names = metadata.get("feature_names")

            # Restore label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(metadata["label_encoder_classes"])
            
            # Restore categorical encoders
            self.categorical_encoders = {}
            for col, classes in metadata.get("categorical_encoders", {}).items():
                encoder = LabelEncoder()
                encoder.classes_ = np.array(classes)
                self.categorical_encoders[col] = encoder

            print(f"🧠 Metadata loaded from: {metadata_path}")
            
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {metadata_path}")
        except Exception as e:
            print(f"Warning: Error loading metadata: {e}")