import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def featurize_molecules():
    print("Loading raw dataset...")
    # Load the file we generated in the last step
    df = pd.read_csv("malaria_raw.csv")
    
    print("Converting SMILES strings into mathematical fingerprints...")
    fingerprints = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles_string = row['smiles']
        # Turn text into an active RDKit Molecule object
        mol = Chem.MolFromSmiles(smiles_string)
        
        if mol is not None:
            # Generate a Morgan Fingerprint (radius 2, length 1024 bits)
            # This turns the 3D chemistry structure into a list of 1024 zeros and ones
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            # Convert the bit vector to a simple Python list of integers
            fp_as_list = list(fp)
            fingerprints.append(fp_as_list)
            valid_indices.append(idx)
        else:
            print(f"Warning: Invalid SMILES skipped at row {idx}")

    # Create a clean DataFrame containing only the valid processed molecules
    processed_df = df.iloc[valid_indices].copy()
    
    # Save the processed math features into a new CSV
    # Each row will have its ID, structural data, and its binary activity score
    processed_df['fingerprint'] = fingerprints
    processed_df.to_csv("malaria_processed.csv", index=False)
    
    print("\nFeaturization complete!")
    print(f"Successfully processed {len(processed_df)} molecules.")
    print("Vector length for each molecule: 1024 dimensions.")
    print("Saved numeric feature dataset as 'malaria_processed.csv'")

if __name__ == "__main__":
    featurize_molecules()