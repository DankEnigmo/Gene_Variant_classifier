import pandas as pd
import re
from typing import Tuple
from pathlib import Path


class VariantPreprocessor:
    def __init__(self, file_path: str):
        """Initialize the preprocessor with the path to the ACMG-annotated variants CSV."""
        self.file_path = Path(file_path)
        self.data = None
        
    def load_data(self) -> None:
        """Load and parse the ACMG-annotated variants dataset."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        self.data = pd.read_csv(self.file_path, sep='\t', low_memory=False)
    
    @staticmethod
    def parse_intervar_string(s: str) -> dict:
        """Parse InterVar string to extract ACMG criteria features."""
        features = {}
        
        # Handle NaN/None values
        if pd.isna(s) or not isinstance(s, str):
            features['acmg_class'] = 'VUS'
            features['PVS1'] = 0
            features['BA1'] = 0
            
            # Initialize all PS, PM, PP, BS, BP criteria as 0
            for key in ['PS', 'PM', 'PP', 'BS', 'BP']:
                max_indices = {'PS': 4, 'PM': 6, 'PP': 5, 'BS': 4, 'BP': 7}
                for i in range(1, max_indices[key] + 1):
                    features[f'{key}{i}'] = 0
            return features
        
        # Extract classification
        match = re.search(r'InterVar:\s*(.*?)\s*PVS1', s)
        features['acmg_class'] = match.group(1).strip() if match else 'VUS'

        # Extract PVS1 and BA1 (single values)
        for key in ['PVS1', 'BA1']:
            m = re.search(rf'{key}=(\d+)', s)
            features[key] = int(m.group(1)) if m else 0

        # Extract array-style fields: PS, PM, PP, BS, BP
        array_keys = ['PS', 'PM', 'PP', 'BS', 'BP']
        max_indices = {'PS': 4, 'PM': 6, 'PP': 5, 'BS': 4, 'BP': 7}
        
        for key in array_keys:
            m = re.search(rf'{key}=\[(.*?)\]', s)
            if m:
                values = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
            else:
                values = []
            
            # Create individual features for each criterion
            for i in range(1, max_indices[key] + 1):
                if i <= len(values):
                    features[f'{key}{i}'] = values[i-1]
                else:
                    features[f'{key}{i}'] = 0

        return features
    
    def extract_acmg_features(self) -> None:
        """Extract ACMG criteria features from InterVar column."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Find the InterVar column
        intervar_col = ' InterVar: InterVar and Evidence '
        if intervar_col not in self.data.columns:
            raise ValueError(f"InterVar column '{intervar_col}' not found in dataset")
        
        print(f"Extracting ACMG features from column: {intervar_col}")
        
        # Apply parsing function to extract ACMG features
        acmg_features = self.data[intervar_col].apply(self.parse_intervar_string)
        
        # Convert to DataFrame and join with original data
        acmg_df = pd.DataFrame(acmg_features.tolist())
        self.data = pd.concat([self.data, acmg_df], axis=1)
        
        print(f"Extracted ACMG features: {list(acmg_df.columns)}")
            
    def clean_data(self) -> None:
        """Clean and preprocess the data for multi-class classification."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Map your actual column names to standardized names
        column_mapping = {
            '#Chr': 'chromosome',
            'Start': 'start',
            'End': 'end', 
            'Ref': 'reference',
            'Alt': 'alternate',
            'Ref.Gene': 'gene',
            'Func.refGene': 'func_gene',
            'ExonicFunc.refGene': 'exonic_func',
            'Freq_gnomAD_genome_ALL': 'freq_gnomad_genome_all',
            'CADD_phred': 'cadd_phred',
            'SIFT_score': 'sift_score',
            'Polyphen2_HDIV_score': 'polyphen2_hdiv_score',
            'GERP++_RS': 'gerp_rs',
            'clinvar: Clinvar': 'clinvar'
        }
        
        # Rename columns that exist
        existing_mappings = {old: new for old, new in column_mapping.items() if old in self.data.columns}
        self.data.rename(columns=existing_mappings, inplace=True)
        
        # Extract ACMG features from InterVar string
        self.extract_acmg_features()

        # Use the extracted classification as the main classification
        self.data['classification'] = self.data['acmg_class']
        
        # Standardize classification labels to match XGBoost model expectations
        classification_mapping = {
            'Uncertain significance': 'VUS',
            'UNK': 'VUS',
            'Likely pathogenic': 'Likely_pathogenic',
            'Likely benign': 'Likely_benign',
            'Pathogenic': 'Pathogenic',
            'Benign': 'Benign'
        }
        
        self.data['classification'] = self.data['classification'].replace(classification_mapping).fillna('VUS')

        # Compute variant length
        self.data['start'] = pd.to_numeric(self.data['start'], errors='coerce')
        self.data['end'] = pd.to_numeric(self.data['end'], errors='coerce')
        self.data['variant_length'] = self.data['end'] - self.data['start']
        
        # Handle reference and alternate allele lengths
        self.data['ref_length'] = self.data['reference'].astype(str).str.len()
        self.data['alt_length'] = self.data['alternate'].astype(str).str.len()
        self.data['is_indel'] = (self.data['ref_length'] != self.data['alt_length']).astype(int)

        # Convert and clean numeric columns
        numeric_columns = {
            'sift_score': 0.5,
            'polyphen2_hdiv_score': 0.5,
            'cadd_phred': 0.0,
            'gerp_rs': 0.0,
            'freq_gnomad_genome_all': 0.0
        }
        
        for col, default in numeric_columns.items():
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(default)

        # Drop rows with missing essential information
        essential_cols = ['chromosome', 'start', 'end', 'reference', 'alternate', 'gene', 'classification']
        self.data.dropna(subset=essential_cols, inplace=True)
        
        # Convert chromosome to string and handle potential formatting issues
        self.data['chromosome'] = self.data['chromosome'].astype(str).str.replace('chr', '', case=False)
        
        print(f"Data shape after cleaning: {self.data.shape}")
        print(f"Classification distribution:\n{self.data['classification'].value_counts()}")
        
    def extract_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for ML model including ACMG criteria."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Base genomic features
        base_features = [
            'chromosome', 'start', 'end', 'gene',
            'variant_length', 'ref_length', 'alt_length', 'is_indel'
        ]
        
        # Functional prediction scores
        score_features = []
        for col in ['sift_score', 'polyphen2_hdiv_score', 'cadd_phred', 
                   'gerp_rs', 'freq_gnomad_genome_all']:
            if col in self.data.columns:
                score_features.append(col)
        
        # ACMG criteria features
        acmg_features = ['PVS1', 'BA1']
        for prefix in ['PS', 'PM', 'PP', 'BS', 'BP']:
            max_indices = {'PS': 4, 'PM': 6, 'PP': 5, 'BS': 4, 'BP': 7}
            for i in range(1, max_indices[prefix] + 1):
                feature_name = f'{prefix}{i}'
                if feature_name in self.data.columns:
                    acmg_features.append(feature_name)
        
        # Combine all features
        features = base_features + score_features + acmg_features
        
        # Check for missing feature columns
        available_features = [col for col in features if col in self.data.columns]
        missing_features = [col for col in features if col not in self.data.columns]
        
        if missing_features:
            print(f"Warning: Missing feature columns: {missing_features}")
        
        print(f"Using {len(available_features)} features: {available_features}")

        X = self.data[available_features]
        y = self.data['classification']

        return X, y
    
    def get_acmg_summary(self) -> pd.DataFrame:
        """Get summary statistics of ACMG criteria usage."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        acmg_cols = ['PVS1', 'BA1']
        for prefix in ['PS', 'PM', 'PP', 'BS', 'BP']:
            max_indices = {'PS': 4, 'PM': 6, 'PP': 5, 'BS': 4, 'BP': 7}
            for i in range(1, max_indices[prefix] + 1):
                acmg_cols.append(f'{prefix}{i}')
        
        available_cols = [col for col in acmg_cols if col in self.data.columns]
        summary = self.data[available_cols].sum().sort_values(ascending=False)
        
        return pd.DataFrame({
            'ACMG_Criteria': summary.index,
            'Usage_Count': summary.values,
            'Percentage': (summary.values / len(self.data) * 100).round(2)
        })