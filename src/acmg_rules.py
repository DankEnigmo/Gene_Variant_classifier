import os
from pathlib import Path
import pandas as pd
import subprocess

class ACMGRules:
    """
    Class for ACMG-AMP variant classification using ANNOVAR and InterVar.
    """
    def __init__(self):
        self.annovar_path = Path("/annovar")
        self.intervar_path = Path("/InterVar")
        self.data_dir = Path("/app/data")
        
        self.pathogenic_criteria = {
            'PVS1', 'PS1', 'PS2', 'PS3', 'PS4', 'PM1', 'PM2', 'PM3', 'PM4', 'PM5', 'PM6',
            'PP1', 'PP2', 'PP3', 'PP4', 'PP5'
        }
        self.benign_criteria = {
            'BA1', 'BS1', 'BS2', 'BS3', 'BS4', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7'
        }
        self._verify_paths()

    def _verify_paths(self):
        """Verify that ANNOVAR and its databases are properly set up."""
        if not self.annovar_path.exists():
            raise FileNotFoundError(f"ANNOVAR not found at {self.annovar_path}")
        humandb_path = self.annovar_path / "humandb"
        if not humandb_path.exists():
            raise FileNotFoundError(f"ANNOVAR database directory not found at {humandb_path}")
        required_dbs = [
            "hg19_refGene.txt",
            "hg19_esp6500siv2_all.txt",
            "hg19_avsnp147.txt",
            "hg19_dbnsfp42a.txt",
            "hg19_clinvar_20221231.txt",
            "hg19_gnomad_genome.txt",
            "hg19_dbscsnv11.txt"
        ]
        missing_dbs = [db for db in required_dbs if not (humandb_path / db).exists()]
        if missing_dbs:
            raise FileNotFoundError(f"Missing required ANNOVAR databases: {', '.join(missing_dbs)}")

    def annotate_vcf(self, vcf_file: str) -> str:
        """Annotate VCF file using ANNOVAR and InterVar."""
        input_file = self.data_dir / "test.txt"
        output_base = self.data_dir / "test_annovar"
        annotated_file = f"{output_base}.hg19_multianno.txt"
        intervar_file = f"{annotated_file}.intervar"

        if Path(intervar_file).exists():
            print(f"[INFO] Skipping annotation steps — InterVar output already exists at {intervar_file}")
            return annotated_file

        # Run ANNOVAR: convert VCF to input format
        if not input_file.exists():
            subprocess.run([
                "perl", str(self.annovar_path / "convert2annovar.pl"),
                "-format", "vcf4", str(vcf_file), "-outfile", str(input_file)
            ], check=True)
        else:
            print(f"[INFO] Skipping convert2annovar — input file already exists at {input_file}")

        # Run ANNOVAR: annotate the input file
        if not Path(annotated_file).exists():
            subprocess.run([
                "perl", str(self.annovar_path / "table_annovar.pl"),
                str(input_file), str(self.annovar_path / "humandb"),
                "-buildver", "hg19", "-out", str(output_base),
                "-protocol", "refGene,esp6500siv2_all,avsnp147,dbnsfp42a,clinvar_20221231,gnomad_genome,dbscsnv11",
                "-operation", "g,f,f,f,f,f,f", "-nastring", "."
            ], check=True)
        else:
            print(f"[INFO] Skipping table_annovar — annotated file already exists at {annotated_file}")

        # Run InterVar: classify the annotated output
        subprocess.run([
            "python3", str(self.intervar_path / "intervar.py"),
            "-i", annotated_file,
            "-o", str(output_base),
            "-d", str(self.intervar_path / "intervardb"),  # Add this line
            "--buildver", "hg19"
        ], check=True)

        return annotated_file


    def evaluate_variant(self, row):
        """Apply ACMG-AMP rules to a single variant row."""
        sift = float(row.get('SIFT_score', 0.5))
        polyphen = float(row.get('Polyphen2_HDIV_score', 0.5))
        af = float(row.get('AF', 0.0))
        exonic_func = row.get('ExonicFunc.refGene', '')
        intervar_evidence = row.get('InterVar_evidence', '')
        gene = row.get('Gene.refGene', 'Unknown')
        pathogenic, benign = set(), set()
        if exonic_func in ['frameshift deletion', 'frameshift insertion', 'stopgain', 'stoploss']:
            pathogenic.add('PVS1')
        if af < 0.0001:
            pathogenic.add('PM2')
        if sift < 0.05 and polyphen > 0.908:
            pathogenic.add('PP3')
        if af > 0.05:
            benign.add('BA1')
        if sift > 0.05 and polyphen < 0.447:
            benign.add('BP4')
        if pd.notna(intervar_evidence):
            for ev in str(intervar_evidence).split(','):
                ev = ev.strip()
                if ev in self.pathogenic_criteria:
                    pathogenic.add(ev)
                elif ev in self.benign_criteria:
                    benign.add(ev)
        classification = self.classify(pathogenic, benign)
        confidence = self.confidence_score(pathogenic, benign)
        return pd.Series({
            'classification': classification,
            'confidence': confidence,
            'pathogenic_evidence': ','.join(sorted(pathogenic)) or 'None',
            'benign_evidence': ','.join(sorted(benign)) or 'None',
            'SIFT_score': sift,
            'Polyphen2_HDIV_score': polyphen,
            'AF': af,
            'ExonicFunc.refGene': exonic_func,
            'InterVar_evidence': intervar_evidence,
            'gene': gene
        })

    def classify(self, pathogenic, benign):
        p, b = len(pathogenic), len(benign)
        if 'PVS1' in pathogenic or p >= 4:
            return "Pathogenic"
        elif p >= 3:
            return "Likely Pathogenic"
        elif 'BA1' in benign or b >= 2:
            return "Benign"
        elif b >= 1:
            return "Likely Benign"
        return "Uncertain Significance"

    def confidence_score(self, pathogenic, benign):
        p, b = len(pathogenic), len(benign)
        if 'PVS1' in pathogenic or p >= 4:
            return 1.0
        elif p >= 3:
            return 0.8
        elif 'BA1' in benign or b >= 2:
            return 1.0
        elif b >= 1:
            return 0.8
        return 0.5

    def evaluate_variants(self, vcf_file: str) -> pd.DataFrame:
        """Annotate and classify variants from a VCF file."""
        annotated_file = self.annotate_vcf(vcf_file)
        df = pd.read_csv(annotated_file, sep='\t', low_memory=False)
        results = df.apply(self.evaluate_variant, axis=1)
        final = pd.concat([
            df[['Chr', 'Start', 'End', 'Ref', 'Alt']].rename(
                columns={
                    'Chr': 'chromosome', 'Start': 'start', 'End': 'end',
                    'Ref': 'reference', 'Alt': 'alternate'
                }
            ),
            results
        ], axis=1)
        return final

def main():
    acmg = ACMGRules()
    intervar_file = acmg.data_dir / "test_annovar.hg19_multianno.txt.intervar"
    tsv_output = acmg.data_dir / "test.tsv"

    # Skip annotation if InterVar file already exists
    if not intervar_file.exists():
        vcf_file = acmg.data_dir / "clinvar.vcf.gz"
        if not vcf_file.exists():
            print(f"[ERROR] VCF file not found at {vcf_file}")
            return
        print(f"[INFO] InterVar output not found, running annotation...")
        intervar_file = acmg.annotate_vcf(str(vcf_file))
    else:
        print(f"[INFO] Using existing InterVar output at {intervar_file}")

    # Convert InterVar file to TSV
    try:
        if not tsv_output.exists():
            df = pd.read_csv(intervar_file, sep='\t', low_memory=False)
            tsv_output = acmg.data_dir / "intervar_final_output.tsv"
            df.to_csv(tsv_output, sep='\t', index=False)
            print(f"[SUCCESS] InterVar TSV file created at: {tsv_output}")
        else:
            print(f"[INFO] Using existing InterVar TSV file at {tsv_output}")
    except Exception as e:
        print(f"[ERROR] Failed to process InterVar file: {e}")


if __name__ == "__main__":
    main() 