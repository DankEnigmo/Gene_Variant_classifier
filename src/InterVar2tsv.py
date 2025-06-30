import pandas as pd
from pathlib import Path

def convert_intervar_to_tsv(intervar_file_path: str, output_tsv_path: str):
    intervar_file = Path(intervar_file_path)
    output_file = Path(output_tsv_path)

    if not intervar_file.exists():
        print(f"[ERROR] InterVar file not found at {intervar_file}")
        return

    try:
        print(f"[INFO] Reading: {intervar_file}")
        df = pd.read_csv(intervar_file, sep='\t', low_memory=False)

        print(f"[INFO] Saving as TSV: {output_file}")
        df.to_csv(output_file, sep='\t', index=False)
        print(f"[SUCCESS] Saved to: {output_file}")

    except Exception as e:
        print(f"[ERROR] Failed to convert file: {e}")

# Example usage
if __name__ == "__main__":
    intervar_path = r"D:\Genomiki\data\annovar_test.hg19_multianno.txt.intervar"
    tsv_output_path = r"D:\Genomiki\data\test.tsv"
    convert_intervar_to_tsv(intervar_path, tsv_output_path)
