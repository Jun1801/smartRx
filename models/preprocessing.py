import os
import itertools
import csv
import sys 
import pickle

import pandas as pd
import numpy as np
import warnings


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from contextlib import contextmanager

from tqdm import tqdm


def parse_food_input(input_file: str) -> None:
    """
    Phân tích file chứa danh sách thuốc và thực phẩm, truy xuất thông tin SMILES và CAS number
    từ các file lookup, sau đó lưu kết quả vào file CSV theo định dạng: Prescription, Drug name, Smiles

    Tham số:
        input_file: Đường dẫn đến file chứa danh sách thuốc và thực phẩm, mỗi dòng phân tách bằng tab và dạng 'tên|mã'
    """

    food_compound = pd.read_csv('data/Dataset/food_compounds_lookup.csv')
    drug_info = pd.read_csv('data/Dataset/drug_info_combined.csv')

    all_drugs = []
    all_foods = []

    with open(input_file, 'r') as fp:
        for idx, line in enumerate(fp):
            items = [i.strip().lower() for i in line.strip().split('\t')]
            if idx == 0:
                all_drugs.extend(items)
            else:
                all_foods.extend(items)

    assert all_drugs and all_foods, 'No valid drug-food pairs entered.'

    with open('data/Result/parsed_input.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prescription', 'Drug name', 'Smiles'])

        count = 0
        for drug_entry in all_drugs:
            try:
                drug_name, drug_compound = drug_entry.split('|')
                drug_row = drug_info.loc[drug_info['Name'].str.lower() == drug_compound.lower()]

                if drug_row.empty:
                    print(f"No match found for {drug_compound}")
                    continue

                drug_smiles = drug_row['Smiles'].values[0]
                drug_display = f"{drug_name}({drug_compound})"

                for food_entry in all_foods:
                    try:
                        food_name, food_compound_code = food_entry.split('|')
                        food_row = food_compound.loc[food_compound['name'].str.lower() == food_compound_code.lower()]

                        if food_row.empty:
                            print(f"No match found for {food_compound_code}")
                            continue

                        food_cas = food_row['cas_number'].values[0]
                        food_display = f"{food_name}({food_compound_code})"

                        writer.writerow([count, drug_display, drug_smiles])
                        writer.writerow([count, food_display, food_cas])
                        count += 1

                    except:
                        continue

            except:
                continue

def parse_drug_input(input_file: str) -> None:
    """
    Phân tích file chứa danh sách thuốc, truy xuất thông tin SMILES và xuất ra file CSV
    theo từng cặp thuốc chính và phụ.

    Tham số:
        input_file: Đường dẫn tới file chứa 2 dòng, mỗi dòng là danh sách thuốc cách nhau bằng tab, theo định dạng 'tên|mã'.
    """

    drug_info = pd.read_csv('data/Dataset/drug_info_combined.csv')
    drug_info['Name_lower'] = drug_info['Name'].str.lower()

    with open(input_file, 'r') as f:
        lines = f.readlines()

    current_drugs = [i.strip() for i in lines[0].strip().split('\t')]
    other_drugs = [i.strip() for i in lines[1].strip().split('\t')]

    output_path = 'data/Result/parsed_drug_input.csv'
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prescription', 'Drug name', 'Smiles'])

        count = 0
        for drug_i in current_drugs:
            try:
                name_i, compound_i = drug_i.lower().split('|')
                match_i = drug_info.loc[drug_info['Name_lower'] == compound_i]
                if match_i.empty:
                    print(f"No match found for {compound_i}")
                    continue
                smile_i = match_i['Smiles'].values[0]
                drug_display_i = f"{name_i}({compound_i})"
            except:
                continue

            for drug_j in other_drugs:
                try:
                    name_j, compound_j = drug_j.lower().split('|')
                    match_j = drug_info.loc[drug_info['Name_lower'] == compound_j]
                    if match_j.empty:
                        print(f"No match found for {compound_j}")
                        continue
                    smile_j = match_j['Smiles'].values[0]
                    drug_display_j = f"{name_j}({compound_j})"
                except:
                    continue

                writer.writerow([count, drug_display_i, smile_i])
                writer.writerow([count, drug_display_j, smile_j])
                count += 1

def parse_DDI_input_file(input_file: str, output_file: str) -> None:
    """
    Đọc file CSV chứa danh sách thuốc theo đơn và sinh tất cả các cặp thuốc có thể có kèm SMILES,
    rồi lưu vào file CSV theo định dạng: Prescription, Drug1, Drug1_SMILES, Drug2, Drug2_SMILES

    Tham số:
        input_file: File CSV chứa các cột Prescription, Drug name, Smiles.
        output_file: File CSV đầu ra chứa các cặp thuốc và SMILES.
    """
    drug_pair_info = {}
    drug_smiles_info = {}

    # Đọc file CSV bằng pandas
    df = pd.read_csv(input_file)

    for _, row in df.iterrows():
        prescription = str(row['Prescription']).strip()
        drug_name = str(row['Drug name']).strip()
        smiles = str(row['Smiles']).strip()

        drug_smiles_info[(prescription, drug_name)] = smiles
        drug_pair_info.setdefault(prescription, []).append(drug_name)

    # Ghi file kết quả
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prescription', 'Drug1', 'Drug1_SMILES', 'Drug2', 'Drug2_SMILES'])

        for each_prescription in drug_pair_info:
            drug_names = drug_pair_info[each_prescription]
            for drug1, drug2 in itertools.combinations(drug_names, 2):
                drug1_smiles = drug_smiles_info[(each_prescription, drug1)]
                drug2_smiles = drug_smiles_info[(each_prescription, drug2)]
                writer.writerow([each_prescription, drug1, drug1_smiles, drug2, drug2_smiles])
                
def _read_molecule(file_path: str) -> rdkit.Chem.Mol|None:
    """
    Đọc phân tử từ file có định dạng SMILES (.smi), MOL (.mol), SDF (.sdf)

    Tham số:
        file_path: Đường dẫn tới file phân tử.

    Trả về:
        Đối tượng mol từ RDKit hoặc None nếu không hợp lệ.
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.smi':
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            return Chem.MolFromSmiles(line.split()[0])
    elif ext in ['.mol', '.mol2']:
        return Chem.MolFromMolFile(file_path)
    elif ext == '.sdf':
        suppl = Chem.SDMolSupplier(file_path)
        return suppl[0] if suppl and suppl[0] is not None else None
    else:
        return None
        
def _calculate_fingerprint(mol: rdkit.Chem.Mol) -> rdkit.DataStructs.ExplicitBitVect:
    """
    Tính fingerprint ECFP4 dưới dạng vector nhị phân từ một phân tử RDKit.

    Tham số:
        mol: Đối tượng phân tử RDKit.

    Trả về:
        Vector fingerprint dạng nhị phân (Morgan fingerprint).
    """
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def calculate_drug_similarity(drug_dir: str,
                              input_dir: str,
                              output_file: str) -> None:
    """
    Tính độ tương đồng Tanimoto giữa tất cả thuốc trong 2 thư mục chứa file phân tử.

    Tham số:
        drug_dir: Thư mục chứa các file thuốc đầu vào.
        input_dir: Thư mục chứa các file cần so sánh.
        output_file: File CSV để lưu kết quả độ tương đồng.
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
    for drug_name, drug_fp in drug_mols.items():
        for input_name, input_fp in input_mols.items():
            sim = DataStructs.TanimotoSimilarity(drug_fp, input_fp)
            results.append({
                'Drug': drug_name,
                'Input': input_name,
                'TanimotoSimilarity': sim
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


# 2. Tính độ tương đồng giữa danh sách thuốc và một file duy nhất
@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def calculate_structure_similarity(drug_dir: str,
                                   input_file: str,
                                   output_file: str,
                                   drug_list_file: str) -> None:
    # Đọc danh sách ID thuốc cần giữ (trong trường hợp cần đối chiếu)
    with open(drug_list_file, "r") as f:
        drug_list = [line.strip() for line in f if line.strip()]

    # Danh sách file .mol/.sdf
    drugbank_files = [
        os.path.join(drug_dir, f)
        for f in os.listdir(drug_dir)
        if os.path.isfile(os.path.join(drug_dir, f))
    ]

    # Đọc dữ liệu tương tác
    df_input = pd.read_csv(input_file)
    all_input_drug_info = {}

    for _, row in df_input.iterrows():
        drug1 = str(row['Drug1']).strip()
        smiles1 = str(row['Drug1_SMILES']).strip()
        drug2 = str(row['Drug2']).strip()
        smiles2 = str(row['Drug2_SMILES']).strip()

        if drug1 and drug1 not in all_input_drug_info:
            all_input_drug_info[drug1] = smiles1
        if drug2 and drug2 not in all_input_drug_info:
            all_input_drug_info[drug2] = smiles2

    drug_similarity_info = {}

    for input_drug_id, smiles in all_input_drug_info.items():
        try:
            with suppress_stderr():
                mol2 = Chem.MolFromSmiles(smiles)
                if mol2 is None:
                    continue
                mol2 = AllChem.AddHs(mol2)
                fps2 = AllChem.GetMorganFingerprint(mol2, 2)
        except:
            continue

        drug_similarity_info[input_drug_id] = {}

        for drug_path in drugbank_files:
            drugbank_id = os.path.basename(drug_path).split('.')[0]

            try:
                with suppress_stderr():
                    mol1 = Chem.MolFromMolFile(drug_path)
                    if mol1 is None:
                        continue
                    mol1 = AllChem.AddHs(mol1)
                    fps1 = AllChem.GetMorganFingerprint(mol1, 2)

                score = DataStructs.TanimotoSimilarity(fps1, fps2)
                drug_similarity_info[input_drug_id][drugbank_id] = score
            except:
                continue
                # Tạo DataFrame
    df = pd.DataFrame.from_dict(drug_similarity_info, orient='index')

    # Đảm bảo tất cả các cột từ drug_list đều có mặt (cột thiếu sẽ được điền NaN)
    for drug_id in drug_list:
        if drug_id not in df.columns:
            df[drug_id] = None

    # Đảm bảo cột theo đúng thứ tự drug_list
    df = df[drug_list]

    # Xuất ra file CSV
    df.to_csv(output_file)

# 3. Áp dụng PCA lên kết quả độ tương đồng
def calculate_pca(similarity_profile_file: str,
                  output_file: str,
                  pca_model_path: str) -> None:
    """
    Áp dụng PCA lên ma trận độ tương đồng để giảm số chiều và lưu kết quả vào file CSV.

    Tham số:
        similarity_profile_file: Đường dẫn đến file chứa ma trận độ tương đồng.
        output_file: File CSV đầu ra chứa dữ liệu PCA.
        pca_model: File PKL chứa mô hình PCA đã huấn luyện (scikit-learn PCA object).
    """

    # Load PCA model
    with open(pca_model_path, 'rb') as f:
        pca_model = pickle.load(f)

    df = pd.read_csv(similarity_profile_file)
    if df.shape[0] == 0:
        return

    # Lấy dữ liệu số và thay NaN = 0
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0.0)

    # Kiểm tra số chiều khớp với PCA model
    if numeric_df.shape[1] != pca_model.n_features_:
        raise ValueError(f"PCA input mismatch: expected {pca_model.n_features_} features, got {numeric_df.shape[1]}.")

    reduced_data = pca_model.transform(numeric_df)

    # Kết hợp lại
    pca_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    final_df = pd.concat([non_numeric_df.reset_index(drop=True), pca_df], axis=1)

    final_df.to_csv(output_file, index=False)

def generate_input_profile(input_file: str,
                           pca_profile_file: str) -> pd.DataFrame:
    """
    Kết hợp vector PCA của từng cặp thuốc từ đơn thuốc thành một vector đặc trưng đầu vào cho mô hình DNN.

    Tham số:
        input_file: File CSV chứa danh sách các cặp thuốc (Prescription, Drug1, Drug1_SMILES, Drug2, Drug2_SMILES).
        pca_profile_file: File chứa vector PCA của từng thuốc (drug_id làm chỉ mục).
        output_file: Đường dẫn lưu kết quả đầu ra (CSV).

    Trả về:
        DataFrame gồm các vector đặc trưng đầu vào cho từng cặp thuốc.
    """
    # Load PCA profile (chỉ mục là drug_id)
    df_pca = pd.read_csv(pca_profile_file, index_col=0)

    # Load danh sách cặp thuốc từ input CSV
    df_input = pd.read_csv(input_file)

    interaction_list = []
    for _, row in df_input.iterrows():
        prescription = str(row['Prescription']).strip()
        drug1 = str(row['Drug1']).strip()
        drug2 = str(row['Drug2']).strip()

        # Chỉ thêm vào nếu cả hai thuốc có PCA vector
        if drug1 in df_pca.index and drug2 in df_pca.index:
            interaction_list.append([prescription, drug1, drug2])
            interaction_list.append([prescription, drug2, drug1])  # Đảo chiều

    # Lấy các cột PCA
    pca_columns = [col for col in df_pca.columns if col.startswith("PC") or col.startswith("PC_")]
    drug_feature_info = df_pca[pca_columns].to_dict(orient='index')

    ddi_input = {}
    for prescription, drug1, drug2 in tqdm(interaction_list):
        key = f"{prescription}_{drug1}_{drug2}"
        ddi_input[key] = {}

        for col in pca_columns:
            ddi_input[key][f"1_{col}"] = drug_feature_info[drug1][col]
            ddi_input[key][f"2_{col}"] = drug_feature_info[drug2][col]

    # Đảm bảo thứ tự cột 1_PC1..PC50, 2_PC1..PC50
    new_columns = [f"{i}_{col}" for i in [1, 2] for col in pca_columns]
    df_result = pd.DataFrame.from_dict(ddi_input, orient='index')
    df_result = df_result[new_columns]
    # df_result.to_csv("/content/drive/MyDrive/Hackathon 2024 _ SmartRx/Results/output_ddi_pair.csv")

    return df_result
