import os
import itertools
import csv
import sys
import pickle

import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from contextlib import contextmanager
from tqdm import tqdm


def parse_food_input(input_file: str) -> None:
    """
    Parse an input file containing lists of drugs and foods, lookup SMILES and CAS numbers,
    and write the results to a CSV file with columns: Prescription, Drug name, and Smiles or CAS.

    Parameters:
        input_file: Path to the input .txt file. The first line should contain drug entries,
            and subsequent lines contain food entries, each entry tab-separated and formatted as "display_name|compound_code".
    """
    food_compound = pd.read_csv("data/Dataset/food_compounds_lookup.csv")
    drug_info = pd.read_csv("data/Dataset/drug_info_combined.csv")

    all_drugs = []
    all_foods = []

    with open(input_file, "r") as fp:
        for idx, line in enumerate(fp):
            items = [i.strip().lower() for i in line.strip().split("\t")]
            if idx == 0:
                all_drugs.extend(items)
            else:
                all_foods.extend(items)

    assert all_drugs and all_foods, "No valid drug-food pairs entered."

    with open("data/Result/parsed_input.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prescription", "Compound", "Identifier"])

        count = 0
        for drug_entry in all_drugs:
            try:
                display_name, code = drug_entry.split("|")
                drug_row = drug_info.loc[drug_info["Name"].str.lower() == code.lower()]
                if drug_row.empty:
                    print(f"No match found for drug code: {code}")
                    continue

                smiles = drug_row["Smiles"].values[0]
                compound_label = f"{display_name}({code})"
                writer.writerow([count, compound_label, smiles])

                for food_entry in all_foods:
                    try:
                        food_display, food_code = food_entry.split("|")
                        food_row = food_compound.loc[food_compound["name"].str.lower() == food_code.lower()]
                        if food_row.empty:
                            print(f"No match found for food code: {food_code}")
                            continue

                        cas_number = food_row["cas_number"].values[0]
                        food_label = f"{food_display}({food_code})"
                        writer.writerow([count, food_label, cas_number])
                        count += 1

                    except Exception:
                        continue

            except Exception:
                continue


def parse_drug_input(input_file: str) -> None:
    """
    Parse an input file containing two lines of drug entries, retrieve SMILES for each,
    and write paired prescriptions to CSV.

    Parameters:
        input_file: Path to a .txt file with two tab-separated lines:
            - First line: primary drugs formatted as "display_name|compound_code".
            - Second line: secondary drugs formatted similarly.
    """
    drug_info = pd.read_csv("data/Dataset/drug_info_combined.csv")
    drug_info["Name_lower"] = drug_info["Name"].str.lower()

    with open(input_file, "r") as f:
        lines = f.readlines()

    primary = [i.strip() for i in lines[0].split("\t")]
    secondary = [i.strip() for i in lines[1].split("\t")]

    output_path = "data/Result/parsed_drug_input.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prescription", "Compound", "Smiles"])

        count = 0
        for entry_primary in primary:
            try:
                disp_i, code_i = entry_primary.lower().split("|")
                match_i = drug_info.loc[drug_info["Name_lower"] == code_i]
                if match_i.empty:
                    print(f"No match found for drug code: {code_i}")
                    continue
                smiles_i = match_i["Smiles"].values[0]
                label_i = f"{disp_i}({code_i})"
            except Exception:
                continue

            for entry_secondary in secondary:
                try:
                    disp_j, code_j = entry_secondary.lower().split("|")
                    match_j = drug_info.loc[drug_info["Name_lower"] == code_j]
                    if match_j.empty:
                        print(f"No match found for drug code: {code_j}")
                        continue
                    smiles_j = match_j["Smiles"].values[0]
                    label_j = f"{disp_j}({code_j})"
                except Exception:
                    continue

                writer.writerow([count, label_i, smiles_i])
                writer.writerow([count, label_j, smiles_j])
                count += 1


def parse_DDI_input_file(input_file: str, output_file: str) -> None:
    """
    Read a CSV file of prescriptions with drug names and SMILES, generate all possible drug pairs,
    and save to a new CSV with both drugs and their SMILES.

    Parameters:
        input_file: Path to the source CSV with columns ["Prescription", "Drug name", "Smiles"].
        output_file: Path where output CSV will be written with columns
            ["Prescription", "Drug1", "Drug1_SMILES", "Drug2", "Drug2_SMILES"].
    """
    drug_pairs = {}
    smiles_map = {}

    df = pd.read_csv(input_file)
    for _, row in df.iterrows():
        presc = str(row["Prescription"]).strip()
        name = str(row["Drug name"]).strip()
        smi = str(row["Smiles"]).strip()
        smiles_map[(presc, name)] = smi
        drug_pairs.setdefault(presc, []).append(name)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prescription", "Drug1", "Drug1_SMILES", "Drug2", "Drug2_SMILES"])
        for presc, drugs in drug_pairs.items():
            for d1, d2 in itertools.combinations(drugs, 2):
                writer.writerow([presc,
                                 d1,
                                 smiles_map[(presc, d1)],
                                 d2,
                                 smiles_map[(presc, d2)]])


def _read_molecule(file_path: str) -> rdkit.Chem.Mol | None:
    """
    Load a molecule from a file in SMILES (.smi), MOL (.mol/.mol2), or SDF (.sdf) format.

    Parameters:
        file_path: Path to the molecule file.

    Returns:
        rdkit.Chem.Mol or None: RDKit molecule object if loaded successfully, otherwise None.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".smi":
        with open(file_path, "r") as f:
            line = f.readline().strip()
            return Chem.MolFromSmiles(line.split()[0])
    elif ext in [".mol", ".mol2"]:
        return Chem.MolFromMolFile(file_path)
    elif ext == ".sdf":
        suppl = Chem.SDMolSupplier(file_path)
        return suppl[0] if suppl and suppl[0] is not None else None
    return None


def _calculate_fingerprint(mol: rdkit.Chem.Mol) -> DataStructs.ExplicitBitVect:
    """
    Calculate the ECFP4 (Morgan) fingerprint as a binary bit vector for a molecule.

    Parameters:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        rdkit.DataStructs.ExplicitBitVect: Binary fingerprint vector of length 2048.
    """
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def calculate_drug_similarity(drug_dir: str,
                              input_dir: str,
                              output_file: str) -> None:
    """
    Compute the Tanimoto similarity between all molecules in two directories and save results.

    Parameters:
        drug_dir: Path to directory containing reference drug files.
        input_dir: Path to directory containing query molecule files.
        output_file: Path to output CSV file with columns ["Drug", "Input", "TanimotoSimilarity"].
    """
    drug_mols = {}
    for fname in os.listdir(drug_dir):
        mol = _read_molecule(os.path.join(drug_dir, fname))
        if mol:
            drug_mols[fname] = _calculate_fingerprint(mol)

    input_mols = {}
    for fname in os.listdir(input_dir):
        mol = _read_molecule(os.path.join(input_dir, fname))
        if mol:
            input_mols[fname] = _calculate_fingerprint(mol)

    results = []
    for dname, dfp in drug_mols.items():
        for iname, ifp in input_mols.items():
            sim = DataStructs.TanimotoSimilarity(dfp, ifp)
            results.append({"Drug": dname, "Input": iname, "TanimotoSimilarity": sim})

    pd.DataFrame(results).to_csv(output_file, index=False)


@contextmanager
def suppress_stderr():
    """
    Context manager to suppress RDKit (or other) error messages written to stderr.
    """
    with open(os.devnull, "w") as devnull:
        old_err = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_err


def calculate_structure_similarity(drug_dir: str,
                                   input_file: str,
                                   output_file: str,
                                   drug_list_file: str) -> None:
    """
    Compare each input drug (from ID pairs) against a DrugBank directory, compute Morgan fingerprint similarities,
    and output a matrix of similarity scores.

    Parameters:
        drug_dir: Directory containing DrugBank .mol or .sdf files.
        input_file: CSV file path with columns ["Prescription", "Drug1", "Drug1_SMILES", "Drug2", "Drug2_SMILES"].
        output_file: Path to write the similarity matrix CSV, indexed by input drugs.
        drug_list_file: Text file listing DrugBank IDs to include as columns in the output.
    """
    with open(drug_list_file, "r") as f:
        drug_list = [line.strip() for line in f if line.strip()]

    bank_files = [os.path.join(drug_dir, f) for f in os.listdir(drug_dir)
                  if os.path.isfile(os.path.join(drug_dir, f))]

    df_input = pd.read_csv(input_file)
    input_info = {}
    for _, row in df_input.iterrows():
        for col in ["Drug1", "Drug2"]:
            name = str(row[col]).strip()
            smi = str(row[f"{col}_SMILES"]).strip()
            input_info.setdefault(name, smi)

    sim_matrix = {}
    for iname, smi in input_info.items():
        try:
            with suppress_stderr():
                mol_in = Chem.MolFromSmiles(smi)
                if not mol_in:
                    continue
                mol_in = AllChem.AddHs(mol_in)
                fp_in = AllChem.GetMorganFingerprint(mol_in, 2)
        except Exception:
            continue

        sim_matrix[iname] = {}
        for path in bank_files:
            bid = os.path.basename(path).split(".")[0]
            try:
                with suppress_stderr():
                    mol_ref = Chem.MolFromMolFile(path)
                    if not mol_ref:
                        continue
                    mol_ref = AllChem.AddHs(mol_ref)
                    fp_ref = AllChem.GetMorganFingerprint(mol_ref, 2)
                sim_matrix[iname][bid] = DataStructs.TanimotoSimilarity(fp_ref, fp_in)
            except Exception:
                continue

    df_out = pd.DataFrame.from_dict(sim_matrix, orient="index")
    for bid in drug_list:
        if bid not in df_out.columns:
            df_out[bid] = np.nan
    df_out = df_out[drug_list]
    df_out.to_csv(output_file)


def calculate_pca(similarity_profile_file: str,
                  output_file: str,
                  pca_model_path: str) -> None:
    """
    Apply a pretrained PCA model to a similarity matrix to reduce dimensionality and save the result.

    Parameters:
        similarity_profile_file: CSV file path of the similarity matrix (numeric values).
        output_file: Path to write the reduced-dimensionality CSV.
        pca_model_path: Path to a pickle file containing a fitted sklearn PCA object.

    Raises:
        ValueError: If the input feature dimension does not match the PCA model"s expected n_features_.
    """
    with open(pca_model_path, "rb") as f:
        pca = pickle.load(f)

    df = pd.read_csv(similarity_profile_file)
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] != pca.n_features_:
        raise ValueError(f"PCA input mismatch: expected {pca.n_features_} features, got {numeric.shape[1]}.")

    transformed = pca.transform(numeric)
    pca_cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    df_pca = pd.DataFrame(transformed, columns=pca_cols)
    df_nonnum = df.select_dtypes(exclude=[np.number]).reset_index(drop=True)
    pd.concat([df_nonnum, df_pca], axis=1).to_csv(output_file, index=False)


def generate_input_profile(input_file: str,
                           pca_profile_file: str) -> pd.DataFrame:
    """
    Construct feature vectors for each drug-drug interaction pair based on PCA-reduced profiles.

    Parameters:
        input_file: CSV file with columns ["Prescription", "Drug1", "Drug1_SMILES", "Drug2", "Drug2_SMILES"].
        pca_profile_file: CSV file where index is drug IDs and columns are PCA components.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an ordered drug pair, and columns are
            ["1_PC1", ..., "1_PCn", "2_PC1", ..., "2_PCn"] representing each drug"s PCA features.
    """
    df_pca = pd.read_csv(pca_profile_file, index_col=0)
    df_input = pd.read_csv(input_file)

    interactions = []
    for _, row in df_input.iterrows():
        presc = str(row["Prescription"]).strip()
        d1 = str(row["Drug1"]).strip()
        d2 = str(row["Drug2"]).strip()
        if d1 in df_pca.index and d2 in df_pca.index:
            interactions.append((presc, d1, d2))
            interactions.append((presc, d2, d1))

    pca_cols = [col for col in df_pca.columns if col.startswith("PC")]
    features = {}
    for presc, d1, d2 in tqdm(interactions):
        key = f"{presc}_{d1}_{d2}"
        features[key] = {}
        for col in pca_cols:
            features[key][f"1_{col}"] = df_pca.at[d1, col]
            features[key][f"2_{col}"] = df_pca.at[d2, col]

    ordered_cols = [f"{i}_{col}" for i in [1, 2] for col in pca_cols]
    df_features = pd.DataFrame.from_dict(features, orient="index")
    return df_features[ordered_cols]
