# Genomiki ACMG-AMP Variant Classification Pipeline

This project provides a pipeline for classifying genetic variants using the ACMG-AMP guidelines. It integrates ANNOVAR and InterVar for annotation and rule-based classification, and supports machine learning classification using XGBoost.

## Features
- Annotates VCF files using ANNOVAR
- Interprets variants with InterVar
- Classifies variants as Benign, Likely Benign, Likely Pathogenic, Pathogenic, or Uncertain
- Supports XGBoost-based machine learning classification
- Dockerized for easy deployment

## Requirements
- Python 3.8+
- [ANNNOVAR](http://www.openbioinformatics.org/annovar/)
- [InterVar](https://github.com/WGLab/InterVar)
- Docker (for containerized runs)

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd Genomiki
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and set up ANNOVAR and InterVar in the project directory:
   - Place ANNOVAR in `annovar/`
   - Place InterVar in `InterVar/`

## Usage

### Running Locally
1. Prepare your input VCF file and place it in the `data/` directory (e.g., `data/clinvar.vcf.gz`).
2. Run the main script:
   ```bash
   python src/main.py
   ```
3. Results will be saved in `data/variant_classifications.csv`.

### Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t genomiki .
   ```
2. Run the container (replace `D:/Genomiki` with your actual path):
   ```bash
   docker run -it --rm -v "D:/Genomiki/annovar:/mnt/annovar" -v "D:/Genomiki/InterVar:/mnt/InterVar" -v "D:/Genomiki/data:/app/data" genomiki
   ```

## XGBoost Classification
- The pipeline supports XGBoost-based classification to predict the same categories as ACMG-AMP rules: Benign, Likely Benign, Likely Pathogenic, Pathogenic, and Uncertain.
- You can train or use a pre-trained XGBoost model for variant classification. (See `src/` for implementation details.)

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is for research and educational use. Please see individual tool licenses (ANNNOVAR, InterVar) for their terms. 