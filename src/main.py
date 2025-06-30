import os
from pathlib import Path
from acmg_rules import ACMGRules
from preprocessing import VariantPreprocessor
from ml_model import XGBoostMultiClassClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'variant_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def plot_confusion_matrix(y_test, y_pred, class_labels, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Confusion matrix saved")

def save_classification_report(y_test, y_pred, class_labels, output_dir):
    """Generate and save classification report."""
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")
    logging.info("Classification report saved")

def main():
    # Initialize paths
    base_dir = Path(r"D:\Genomiki")
    data_dir = base_dir / "data"
    vcf_file = data_dir / "test.vcf"
    output_file = data_dir / "test.tsv"
    
    logging.info("Starting variant classification pipeline...")
    
    # Step 1: ACMG-AMP Classification
    logging.info("1. Running ACMG-AMP classification...")
    try:
        if not output_file.exists():
            acmg = ACMGRules()
            results = acmg.evaluate_variants(str(vcf_file))
            results.to_csv(output_file, sep='\t', index=False)
            logging.info(f"ACMG-AMP classification completed. Found {len(results)} variants.")
        else:
            results = pd.read_csv(output_file, sep='\t')
            logging.info(f"Loaded existing ACMG-AMP results. Found {len(results)} variants.")
    except Exception as e:
        logging.error(f"Error in ACMG-AMP classification: {e}")
        return
    
    # Step 2: Data Preprocessing for ML
    logging.info("2. Preprocessing data for ML model...")
    try:
        preprocessor = VariantPreprocessor(str(output_file))
        preprocessor.load_data()
        preprocessor.clean_data()
        X, y = preprocessor.extract_features()
        summary_df = preprocessor.get_acmg_summary()

        logging.info("ACMG Criteria Summary:")
        for _, row in summary_df.iterrows():
            logging.info(f"{row['ACMG_Criteria']}: Used {int(row['Usage_Count'])} times ({row['Percentage']}%)")
        logging.info(f"Preprocessing completed. Features shape: {X.shape}")
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return
    
    # Step 3: Train and Evaluate ML Model
    logging.info("3. Training and evaluating ML model...")
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model_path = data_dir / "xgboost_model.pkl"
        
        # Initialize and train model
        model = XGBoostMultiClassClassifier()
        
        logging.info(f"Class distribution (y_train):\n{y_train.value_counts()}")
        
        if model_path.exists():
            logging.info("Found saved model. Loading...")
            model.load_model(str(model_path))
        else:
            logging.info("Training new model...")
            model.train(X_train, y_train, X_test, y_test, tune_hyperparams=True)
            model.save_model(str(model_path))
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Generate additional plots (moved confusion matrix to model.evaluate())
        y_pred, y_proba = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model.class_labels, data_dir)
        save_classification_report(y_test, y_pred, model.class_labels, data_dir)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Make predictions on all data
        logging.info("Making predictions on all data...")
        predictions, probabilities = model.predict(X)
        
        # Prepare final output
        output_data = results.loc[X.index].copy()
        output_data['ml_prediction'] = predictions
        
        # Convert probabilities to a more readable format
        prob_cols = [f'prob_{label}' for label in model.class_labels]
        prob_df = pd.DataFrame(probabilities, columns=prob_cols, index=X.index)
        output_data = pd.concat([output_data, prob_df], axis=1)
        
        # Save final results
        final_output_file = data_dir / "final_predictions.tsv"
        output_data.to_csv(final_output_file, sep='\t', index=False)
        logging.info(f"Final results saved to: {final_output_file}")
        
    except Exception as e:
        logging.error(f"Error in ML model training/evaluation: {e}")
        return
    
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()