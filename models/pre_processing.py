import os
import glob
import pickle
import copy
import argparse
import itertools
import pandas as pd
import time
import pandas as pd
import numpy as np
import csv

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm


def parse_food_input(input_file: str):
    """
    Nhận vào một file chứa danh sách loại thuốc - thực phẩm theo cặp:
    - Trích xuất tên và mã hợp chất từng loại thuốc - thực phẩm.
    - Tra cứu thông tin SMILES của thuốc và CAS number của thực phẩm từ 2 file CSV trong dataset.
    - Ghi kết quả ra file 'parsed_input.csv' theo định dạng:
      Prescription,Drug name,Smiles
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


def parse_drug_input(input_file: str):
    """
    Nhận vào một file chứa danh sách thuốc:
    - Dòng đầu là danh sách thuốc chính (current_drugs)
    - Dòng tiếp theo là danh sách thuốc khác (other_drugs)
    - Truy xuất thông tin SMILES từ drug_info_combined.csv
    - Ghi ra file parsed_input.csv theo format: Prescription, Drug name, Smiles
    """

    drug_info = pd.read_csv('data/Dataset/Drug_info_combined.csv')
    drug_info['Name_lower'] = drug_info['Name'].str.lower()

    with open(input_file, 'r') as f:
        lines = f.readlines()

    current_drugs = [i.strip() for i in lines[0].strip().split('\t')]
    other_drugs = [i.strip() for i in lines[1].strip().split('\t')]

    output_path = 'data/Result/parsed_input.csv'
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


def parse_DDI_input_file(input_file: str, output_file: str):
    """
    Đọc file chứa thông tin thuốc theo đơn, và sinh ra tất cả các cặp thuốc (pairs) 
    trong mỗi đơn, kèm theo SMILES của từng thuốc.
    
    Output: file CSV gồm các cột:
        Prescription, Drug1, Drug1_SMILES, Drug2, Drug2_SMILES
    """
    drug_pair_info = {}
    drug_smiles_info = {}

    with open(input_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            prescription = sptlist[0].strip()
            drug_name = sptlist[1].strip()
            smiles = sptlist[2].strip()
            
            drug_smiles_info[(prescription, drug_name)] = smiles
            drug_pair_info.setdefault(prescription, []).append(drug_name)
        
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prescription', 'Drug1', 'Drug1_SMILES', 'Drug2', 'Drug2_SMILES'])

        for each_prescription in drug_pair_info:
            drug_names = drug_pair_info[each_prescription]
            for drug1, drug2 in itertools.combinations(drug_names, 2):
                drug1_smiles = drug_smiles_info[(each_prescription, drug1)] 
                drug2_smiles = drug_smiles_info[(each_prescription, drug2)] 
                writer.writerow([each_prescription, drug1, drug1_smiles, drug2, drug2_smiles])



# Các hàm calculate

# Đọc phân tử từ file định dạng SMILES, MOL, SDF, ...
def read_molecule(file_path):
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

# Tính fingerprint (ECFP4) dạng vector nhị phân
def calculate_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# 1. Tính độ tương đồng giữa tất cả các file trong 2 thư mục
def calculate_drug_similarity(drug_dir, input_dir, output_file):
    drug_mols = {}
    for fname in os.listdir(drug_dir):
        mol = read_molecule(os.path.join(drug_dir, fname))
        if mol:
            drug_mols[fname] = calculate_fingerprint(mol)

    input_mols = {}
    for fname in os.listdir(input_dir):
        mol = read_molecule(os.path.join(input_dir, fname))
        if mol:
            input_mols[fname] = calculate_fingerprint(mol)

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
def calculate_structure_similarity(drug_dir, input_file, output_file, drug_list):
    """
    Tính toán độ tương đồng cấu trúc (Tanimoto similarity) giữa các thuốc trong input_file (CSV)
    và các thuốc trong thư mục drug_dir. Kết quả là ma trận similarity được lưu thành CSV.

    Parameters:
        - drug_dir (str): Đường dẫn thư mục chứa các file thuốc (.mol/.sdf)
        - input_file (str): File CSV chứa thông tin các cặp thuốc theo đơn
                            (gồm các cột: Prescription, Drug1, Smiles1, Drug2, Smiles2)
        - output_file (str): File CSV đầu ra chứa ma trận độ tương đồng
        - drug_list (List[str]): Danh sách các drugbank_id cần giữ lại trong ma trận
    """
    drugbank_files = [
      os.path.join(drug_dir, f)
      for f in os.listdir(drug_dir)
      if os.path.isfile(os.path.join(drug_dir, f))
      ]

    all_input_drug_info = {}

    # Đọc thông tin từ file CSV input
    df_input = pd.read_csv(input_file)

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
                mol1 = Chem.MolFromMolFile(drug_path)
                if mol1 is None:
                    continue
                mol1 = AllChem.AddHs(mol1)
                fps1 = AllChem.GetMorganFingerprint(mol1, 2)
                score = DataStructs.TanimotoSimilarity(fps1, fps2)
                drug_similarity_info[input_drug_id][drugbank_id] = score
            except:
                continue

    df = pd.DataFrame.from_dict(drug_similarity_info, orient='index')

    # Chỉ giữ lại các cột có trong drug_list và tồn tại trong df.columns
    valid_columns = [col for col in drug_list if col in df.columns]
    df = df[valid_columns]

    df.to_csv(output_file)

# 3. Áp dụng PCA lên kết quả độ tương đồng
def calculate_pca(similarity_profile_file, output_file, pca_model):
     """
    Tính toán PCA trên dữ liệu độ tương đồng và lưu kết quả vào file CSV.

    Parameters:
    - similarity_profile_file: đường dẫn tới file CSV chứa ma trận độ tương đồng.
    - output_file: đường dẫn tới file CSV để lưu kết quả PCA.
    - pca_model_path: đường dẫn tới file .pkl chứa mô hình PCA đã huấn luyện.
    """
    df = pd.read_csv(similarity_profile_file)
    if df.shape[0] == 0:
        return

    numeric_df = df.select_dtypes(include=[np.number])
    reduced_data = pca_model.fit_transform(numeric_df)

    pca_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    final_df = pd.concat([non_numeric_df.reset_index(drop=True), pca_df], axis=1)

    final_df.to_csv(output_file, index=False)



def generate_input_profile(input_file, pca_profile_file):  
    """
        +) Đọc file chứa các cặp thuốc từ đơn thuốc (input_file)
        +) Đọc file vector PCA của từng thuốc (pca_profile_file)
        +) Với mỗi cặp thuốc (drug1, drug2), kết hợp vector PCA của cả 2 thành 1 vector 100 chiều (50+50)
        +) Trả về một Dataframe để đưa vào model 
    """  
    df = pd.read_csv(pca_profile_file, index_col=0)
    # df.index = df.index.map(str)
    
    all_drugs = []
    interaction_list = []
    with open(input_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            prescription = sptlist[0].strip()
            drug1 = sptlist[1].strip()
            drug2 = sptlist[3].strip()
            all_drugs.append(drug1)
            all_drugs.append(drug2)
            if drug1 in df.index and drug2 in df.index:
                interaction_list.append([prescription, drug1, drug2]) # Cặp A - B và Cặp B - A là tương tự 
                interaction_list.append([prescription, drug2, drug1]) 
    
    drug_feature_info = {}
    columns = ['PC_%d' % (i + 1) for i in range(50)]
    for row in df.itertuples():
        drug = row.Index
        feature = []
        drug_feature_info[drug] = {}
        for col in columns:
            val = getattr(row, col)
            feature.append(val)
            drug_feature_info[drug][col] = val
    DDI_input = {}
    for each_drug_pair in tqdm.tqdm(interaction_list):
        prescription = each_drug_pair[0]
        drug1 = each_drug_pair[1]
        drug2 = each_drug_pair[2]
        key = '%s_%s_%s' % (prescription, drug1, drug2)
        
        DDI_input[key] = {}
        for col in columns:
            new_col = '1_%s'%(col)
            DDI_input[key][new_col] = drug_feature_info[drug1][col]
            
        for col in columns:
            new_col = '2_%s'%(col)
            DDI_input[key][new_col] = drug_feature_info[drug2][col]

    new_columns = []
    for i in [1,2]:
        for j in range(1, 51):
            new_key = '%s_PC_%s'%(i, j)
            new_columns.append(new_key)
            
    df = pd.DataFrame.from_dict(DDI_input)
    df = df.T
    df = df[new_columns]
    # df.to_csv(output_file)
    return df

