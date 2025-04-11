from typing import Dict, List, Tuple

import pandas as pd
import tqdm
import re
import csv

pd.set_option('mode.chained_assignment', None)

def read_information_file(information_file: str) -> Dict:
    """
    Đọc file thông tin tương tác và trả về dictionary ánh xạ kiểu tương tác với câu mô tả.

    Tham số:
        information_file: Đường dẫn tới file chứa thông tin. File có định dạng:
                                dòng đầu tiên là header, các dòng sau có cấu trúc:
                                "<interaction_type>\t<sentence>"

    Trả về:
        dict: Dictionary với key là interaction_type (str) và value là câu mô tả (str).
    """
    interaction_info = {}
    fp = open(information_file, 'r')
    fp.readline()  # Bỏ qua header
    for line in fp:
        line_split = line.strip().split('\t')
        interaction_type = line_split[0].strip()[1:]
        sentence = line_split[1].strip()
        interaction_info[interaction_type] = sentence
    fp.close()
    return interaction_info


def read_drug_information(drug_information_file: str) -> Dict:
    """
    Đọc file thông tin thuốc và trả về dictionary ánh xạ drugbank_id với danh sách target tương ứng.

    Tham số:
        drug_information_file: Đường dẫn tới file thông tin thuốc. Mỗi dòng chứa nhiều trường dữ liệu, trong đó lần lượt là:
                                     drugbank_id (str), drugbank_name (str), action (str), pharmacological_action (str), target (str)

    Trả về:
        dict: Dictionary với key là drugbank_id (str) và value là danh sách các target (list[str]).
              Chỉ lưu những dòng có action khác 'None' và target khác 'None'.
    """
    drug_information = {}
    with open(drug_information_file, 'r') as fp:
        for line in fp:
            line_split = line.strip().split('\t')
            drugbank_id = line_split[0].strip()
            action = line_split[7].strip()
            target = line_split[5].strip()

            if action != 'None' and target != 'None':
                if drugbank_id not in drug_information:
                    drug_information[drugbank_id] = [target]
                else:
                    drug_information[drugbank_id].append(target)
    return drug_information

def read_drug_enzyme_information(drug_enzyme_information_file: str) -> Dict:
    """
    Đọc file thông tin enzyme liên quan đến thuốc và trả về dictionary ánh xạ drugbank_id với danh sách uniprot_id tương ứng.

    Tham số:
        drug_enzyme_information_file: Đường dẫn tới file thông tin enzyme. Mỗi dòng chứa:
                                      drugbank_id, uniprot_id, action

    Trả về:
        dict: Dictionary với key là drugbank_id và value là danh sách uniprot_id
              cho những dòng có uniprot_id và action khác 'None'.
    """
    drug_information = {}
    with open(drug_enzyme_information_file, 'r') as fp:
        for line in fp:
            line_split = line.strip().split('\t')
            drugbank_id = line_split[0].strip()
            uniprot_id = line_split[4].strip()
            action = line_split[5].strip()

            if uniprot_id != 'None' and action != 'None':
                if drugbank_id not in drug_information:
                    drug_information[drugbank_id] = [uniprot_id]
                else:
                    drug_information[drugbank_id].append(uniprot_id)

    return drug_information

def read_known_DDI_information(known_DDI_file: str) -> Tuple[Dict, Dict]:
    """
    Đọc file thông tin DDI đã biết và trả về 2 dictionary chứa thông tin các thuốc
    bên trái và bên phải của từng tương tác.

    Tham số:
        known_DDI_file: Đường dẫn tới file thông tin DDI, với định dạng:
                        Dòng đầu tiên là header, các dòng sau có cấu trúc:
                        "<left_drug><right_drug><interaction_type>"

    Trả về:
        tuple: (left_ddi_info, right_ddi_info)
            left_ddi_info (dict): Ánh xạ interaction_type (str) tới danh sách thuốc bên trái (không có duplicates).
            right_ddi_info (dict): Ánh xạ interaction_type (str) tới danh sách thuốc bên phải (không có duplicates).
    """
    left_ddi_info = {}
    right_ddi_info = {}
    with open(known_DDI_file, 'r') as fp:
        fp.readline()  # Bỏ qua header
        for line in fp:
            line_split = line.strip().split('\t')

            left_drug = line_split[0].strip()
            right_drug = line_split[1].strip()
            interaction_type = line_split[2].strip()

            if interaction_type not in left_ddi_info:
                left_ddi_info[interaction_type] = [left_drug]
            else:
                left_ddi_info[interaction_type].append(left_drug)

            if interaction_type not in right_ddi_info:
                right_ddi_info[interaction_type] = [right_drug]
            else:
                right_ddi_info[interaction_type].append(right_drug)

    # Loại bỏ duplicates
    for each_interaction_type in left_ddi_info:
        left_ddi_info[each_interaction_type] = list(set(left_ddi_info[each_interaction_type]))

    for each_interaction_type in right_ddi_info:
        right_ddi_info[each_interaction_type] = list(set(right_ddi_info[each_interaction_type]))

    return left_ddi_info, right_ddi_info

def read_similarity_file(similarity_file: str) -> pd.DataFrame:
    """
    Đọc file CSV chứa bảng ma trận độ tương đồng và trả về DataFrame.
    
    Tham số:
        similarity_file: Đường dẫn tới file CSV, file có index cột.
    
    Trả về:
        pd.DataFrame: DataFrame chứa bảng số liệu tương đồng được đọc từ file.
    """
    similarity_df = pd.read_csv(similarity_file, index_col=0)
    return similarity_df


def read_side_effect_info(df: pd.DataFrame,
                          frequency: int = 10) -> Dict:
    """
    Đọc và tổng hợp thông tin side effects của các thuốc từ DataFrame.

    Tham số:
        df: DataFrame chứa thông tin side effects với các cột 'Drug name', 'SIDE EFFECT', 'MEAN'.
        frequency: Ngưỡng giá trị trung bình (MEAN) để chọn lọc side effect. Mặc định là 10.

    Trả về:
        dict: Dictionary với key là tên thuốc (đã chuyển về chữ thường) và value là chuỗi side effects
              được ghép lại dưới định dạng "SIDE_EFFECT(XX.X%)", phân cách bởi dấu chấm phẩy.
              Nếu không có thông tin, value là None.
    """
    new_df = df[df['MEAN'] >= frequency]
    drug_side_effect_info = {}
    for each_drug, each_df in new_df.groupby('Drug name'):
        string_list = []
        for each_index, each_df in each_df.iterrows():
            side_effect = each_df['SIDE EFFECT']
            mean_frequency = each_df['MEAN']
            string_list.append('%s(%.1f%%)' % (side_effect, mean_frequency))

        drug_side_effect_info[each_drug.lower()] = ';'.join(string_list)

    return drug_side_effect_info
def DDI_result_supplement(input_file: str, output_file: str, interaction_info_file: str) -> None:
    # Đọc dữ liệu từ hai file
    predict_df = pd.read_csv(input_file)
    interaction_df = pd.read_csv(interaction_info_file)

    # Tạo các cột bổ sung
    predict_df["Prescription"] = [int(row.split("_")[0]) for row in predict_df["Drug pair"]]
    predict_df["Drug_pair"] = predict_df["Drug pair"]
    predict_df["DDI_type"] = predict_df["Predicted class"]

    # Chuyển kiểu dữ liệu cho cột type để ánh xạ chính xác
    interaction_df["type"] = interaction_df["type"].astype(int)

    # Ánh xạ từ DDI_type sang câu mô tả tương ứng trong interaction_df
    predict_df["Sentence"] = predict_df["DDI_type"].map(interaction_df.set_index("type")["sentence"])

    # Sắp xếp lại thứ tự cột theo yêu cầu
    final_df = predict_df[["Prescription", "Drug_pair", "DDI_type", "Sentence", "Predicted class", "Score", "STD"]]

    # Lưu file kết quả
    final_df.to_csv(output_file, index=False)


def annotate_DDI_results(DDI_output_file: str,
                         similarity_file: str,
                         known_DDI_file: str,
                         output_file: str,
                         side_effect_information_file: str,
                         model_threshold: float,
                         structure_threshold: float) -> None:
    """
    Ghi annotation cho kết quả dự đoán DDI và xuất ra file .csv chuẩn.

    Các bước:
        - Đọc thông tin DDI, tương đồng cấu trúc, tác dụng phụ.
        - Dự đoán tương tác với confidence cao.
        - Ghi annotation ra file CSV với side effect và các thuốc tương đồng.
    """

    # Giả sử các hàm phụ dưới đây đã được định nghĩa:
    # read_known_DDI_information, read_similarity_file, read_side_effect_info
    left_ddi_info, right_ddi_info = read_known_DDI_information(known_DDI_file)
    similarity_df = read_similarity_file(similarity_file)
    DDI_prediction_df = pd.read_csv(DDI_output_file)  # File output_predict_DDI_combined.csv
    side_effect_df = pd.read_csv(side_effect_information_file, sep='\t')
    drug_side_effect_info = read_side_effect_info(side_effect_df, frequency=10)

    with open(output_file, 'w', newline='') as fp:
        writer = csv.writer(fp)

        # Ghi header vào file CSV
        writer.writerow([
            'Prescription', 'Drug_pair', 'Interaction_type', 'Sentence',
            'DDI_prob', 'DDI_prob_std', 'Confidence_DDI',
            'Side effects (left)', 'Side effects (right)',
            'Similar approved drugs (left)', 'Similar approved drugs (right)',
            'drug1', 'drug2'
        ])

        # Duyệt qua các dòng của DataFrame
        for row in tqdm.tqdm(DDI_prediction_df.itertuples(index=False), total=len(DDI_prediction_df)):
            row_dict = row._asdict()

            prescription = str(row_dict.get('Prescription'))
            drug_pair = row_dict.get('Drug_pair')
            DDI_type = str(row_dict.get('DDI_type'))
            sentence = row_dict.get('Sentence')
            score = float(row_dict.get('Score'))
            std = float(row_dict.get('STD'))
            Confidence_DDI = 0

            # Nếu drug_pair có tiền tố dạng "số_" thì loại bỏ tiền tố đó trước khi tách
            match_prefix = re.match(r'\d+_(.+)', drug_pair)
            if match_prefix:
                rest = match_prefix.group(1)
            else:
                rest = drug_pair

            # Dùng regex để tách thông tin của thuốc trái và thuốc phải
            # Mẫu: phần trái và phải có cấu trúc: "tênthuốc(thanh phần)"
            match = re.match(r'(.+\([^()]+\))_(.+\([^()]+\))', rest)
            if not match:
                print(f"[!] Lỗi định dạng Drug_pair: {drug_pair}")
                continue  # Bỏ qua dòng không khớp định dạng
            left_drug, right_drug = match.groups()

            # Tách lấy thông tin bên trong ngoặc làm drugbank ID hoặc thông tin nhận dạng
            left_comp_match = re.findall(r'\(([^()]+)\)$', left_drug)
            right_comp_match = re.findall(r'\(([^()]+)\)$', right_drug)
            if not left_comp_match or not right_comp_match:
                print(f"[!] Không tìm được thông tin nhận dạng cho: {drug_pair}")
                continue
            left_comp = left_comp_match[0]
            right_comp = right_comp_match[0]

            # Lấy thông tin side effect theo drug_side_effect_info nếu có
            left_drug_side_effect = drug_side_effect_info.get(left_comp, '')
            right_drug_side_effect = drug_side_effect_info.get(right_comp, '')

            # Tính Confidence_DDI: nếu score - (std/2) lớn hơn model_threshold thì Confidence_DDI = 1
            if score - std / 2 > model_threshold:
                Confidence_DDI = 1

            # Lấy danh sách các thuốc tương đồng dựa trên DDI_type từ file known DDI
            left_corresponding_drugs = left_ddi_info.get(DDI_type, [])
            right_corresponding_drugs = right_ddi_info.get(DDI_type, [])

            left_selected_drugs = []
            right_selected_drugs = []

            if left_drug in similarity_df.index and left_corresponding_drugs:
                left_cols = similarity_df.columns.intersection(left_corresponding_drugs)
                left_sim_df = similarity_df.loc[left_drug][left_cols]
                left_selected_drugs = list(left_sim_df[left_sim_df >= structure_threshold].index)

            if right_drug in similarity_df.index and right_corresponding_drugs:
                right_cols = similarity_df.columns.intersection(right_corresponding_drugs)
                right_sim_df = similarity_df.loc[right_drug][right_cols]
                right_selected_drugs = list(right_sim_df[right_sim_df >= structure_threshold].index)

            # Ghi một dòng vào file CSV
            writer.writerow([
                prescription, drug_pair, DDI_type, sentence,
                score, std, Confidence_DDI,
                left_drug_side_effect, right_drug_side_effect,
                ';'.join(left_selected_drugs), ';'.join(right_selected_drugs),
                left_drug, right_drug
            ])

def map_severity(prob: float) -> str:
    """
    Ánh xạ giá trị xác suất thành mức độ nghiêm trọng.
    """
    if prob >= 0.9:
        return "Major"
    elif prob >= 0.7:
        return "Moderate"
    elif prob >= 0.5:
        return "Minor"
    elif prob >= 0.3:
        return "Not severe"
    else:
        return "Unknown"

def summarize_prediction_outcome(result_file: str,
                                 output_file: str,
                                 information_file: str) -> None:
    """
    Tóm tắt kết quả dự đoán bằng cách thay thế mẫu câu với tên các thuốc.

    Quy trình:
        - Đọc file thông tin sentence template dựa trên DDI_class từ file information_file.
        - Đọc file kết quả dự đoán result_file (ở đây là final_output.csv, định dạng CSV với header).
        - Với mỗi dòng, lấy thông tin gồm: Prescription, Drug_pair, Interaction_type, DDI_prob (Score), DDI_prob_std (STD).
        - Tách thông tin trong Drug_pair để lấy tên các thuốc (drug1, drug2) và thay thế placeholder trong template sentence.
        - Ghi kết quả ra output_file với định dạng CSV với các cột: Prescription, Drug_pair, DDI_type, Sentence (sau khi thay thế), Score, STD.

    Tham số:
        result_file: Đường dẫn tới file kết quả dự đoán (ví dụ final_output.csv).
        output_file: Đường dẫn tới file sẽ ghi kết quả tóm tắt (dạng CSV).
        information_file: Đường dẫn tới file chứa thông tin sentence template.
    """
    # Đọc thông tin mẫu câu (template)
    sentence_interaction_info = read_information_file(information_file)
    with open(result_file, 'r', newline='') as fp:
        # Sử dụng csv.DictReader cho file CSV đầu vào (phân cách bằng dấu phẩy)
        reader = csv.DictReader(fp)
        with open(output_file, 'w', newline='') as out_fp:
            # Sử dụng csv.writer với delimiter mặc định là dấu phẩy
            writer = csv.writer(out_fp)
            # Ghi header vào file output
            writer.writerow(['Prescription', 'Drug_pair', 'DDI_type', 'Sentence', 'Final severity' 'Score', 'STD', 'Side_effects (left)', 'Side_effects (right)'])

            for row in reader:
                # Lấy các trường cần thiết từ file kết quả
                prescription = row.get('Prescription', '').strip()
                drug_pair_raw = row.get('Drug_pair', '').strip()
                # Sử dụng cột Interaction_type làm DDI_type (có thể là "Interaction_type" hoặc "DDI_type", tùy theo file)
                DDI_type = row.get('Interaction_type', '').strip()
                Side_effect_left = row.get('Side effects (left)', '').strip()
                Side_effect_right = row.get('Side effects (right)', '').strip()
                score = row.get('DDI_prob', '').strip()  # hoặc "Score" nếu tên cột là Score
                std = row.get('DDI_prob_std', '').strip()  # hoặc "STD"
                try:
                    score_float = float(score)
                except ValueError:
                    print(f"[Warning] Không thể chuyển đổi DDI_prob: {score}")
                    continue

                # Xử lý trường Drug_pair để tách ra tên các thuốc
                # Ví dụ: "1_amoxicillin(amoxicillin)_metformin(metformin)"
                # Bước 1: Loại bỏ tiền tố số nếu có.
                match_prefix = re.match(r'\d+_(.+)', drug_pair_raw)
                if match_prefix:
                    rest = match_prefix.group(1)
                else:
                    rest = drug_pair_raw

                # Bước 2: Dùng regex để tách thành 2 nhóm: left_drug và right_drug
                match = re.match(r'(.+\([^()]+\))_(.+\([^()]+\))', rest)
                if not match:
                    print(f"[Warning] Định dạng Drug_pair không khớp: {drug_pair_raw}")
                    continue
                drug1_full, drug2_full = match.groups()
                # Lấy tên thuốc là phần trước dấu "("
                drug1 = drug1_full.split('(')[0].strip()
                drug2 = drug2_full.split('(')[0].strip()
                # Xây dựng lại Drug_pair chỉ bao gồm tên thuốc (nếu cần)
                drug_pair = f"{drug1}_{drug2}"

                # Lấy mẫu câu từ thông tin tương tác theo DDI_type
                template_sentence = sentence_interaction_info.get(DDI_type, "")
                # Thay thế placeholder #Drug1 và #Drug2 bằng tên thuốc
                prediction_outcome = template_sentence.replace('#Drug1', drug1).replace('#Drug2', drug2)[:-5]
                severity = map_severity(score_float)
               
                # Ghi dòng kết quả ra file output (CSV)
                writer.writerow([prescription, drug_pair, DDI_type, prediction_outcome, severity, score, std, Side_effect_left, Side_effect_right])

def processing_network(df: pd.DataFrame,
                       type_df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý và gán thông tin hành động (action), người gây tác động (perpetrator) và nạn nhân (victim)
    cho các dòng có Interaction_type thuộc danh sách các loại PK (pharmacokinetic).

    Tham số:
        df: DataFrame chứa thông tin DDI cần xử lý.
        type_df: DataFrame chứa thông tin mapping giữa type (loại tương tác) với
                 action và perpetrator (các cột bao gồm 'type', 'action', 'perpetrator').

    Trả về:
        pd.DataFrame: DataFrame được bổ sung thêm các cột 'action', 'perpetrator', 'victim' dựa trên mapping.
    """
    PK_type_list = list(type_df['type'])
    type_to_action = {}
    type_to_perpetrator = {}
    for row in type_df.itertuples():
        type_num = getattr(row, 'type')
        action = getattr(row, 'action')
        perpet = getattr(row, 'perpetrator')
        type_to_action[type_num] = action
        type_to_perpetrator[type_num] = perpet
    PK_df = df[df['Interaction_type'].isin(PK_type_list)]
    action_list = []
    perpet_list = []
    victim_list = []
    for row in PK_df.itertuples():
        type_num = getattr(row, 'Interaction_type')
        drug1 = getattr(row, 'drug1')
        drug2 = getattr(row, 'drug2')
        action = type_to_action[type_num]
        perpet = type_to_perpetrator[type_num]
        action_list.append(action)
        if perpet == '#Drug2':
            perpet_list.append(drug2)
            victim_list.append(drug1)
        else:
            perpet_list.append(drug1)
            victim_list.append(drug2)
    PK_df['action'] = action_list
    PK_df['perpetrator'] = perpet_list
    PK_df['victim'] = victim_list
    return PK_df

def _get_unidirectional_pred(tmp: Dict) -> Dict:
    """
    Tìm hướng dự đoán duy nhất dựa vào dictionary chứa các dự đoán conflict.

    Tham số:
        tmp: Dictionary có cấu trúc {key: (direction, score)} với score là số thực.

    Trả về:
        dict: Nếu tìm thấy hướng duy nhất (direction) với score lớn nhất, trả về subset của tmp với
              các phần tử có cùng direction đó. Nếu không, trả về dictionary rỗng.
    """
    direction = None
    max_key = None
    standard = -float('inf')
    for key, val in tmp.items():
        if val[1] > standard:
            standard = val[1]
            max_key = key
            direction = val[0]
        elif val[1] == standard and direction != val[0]:
            max_key = None

    if max_key is None:
        return {}
    else:
        return {key: val for key, val in tmp.items() if val[0] == direction}

def find_conflicts(df: pd.DataFrame) -> List[int]:
    """
    Tìm các cặp dự đoán xung đột trong DataFrame dựa trên thông tin drug pair và action.

    Quy trình:
        - Xác định các dòng có drug pair (theo thứ tự) đã được báo cáo và ghi nhận thông tin dự đoán.
        - Nếu một drug pair được báo cáo với các giá trị dự đoán khác nhau (action khác nhau), lưu lại.
        - Sử dụng hàm _get_unidirectional_pred để lọc kết quả nếu có hướng dự đoán nhất định.

    Tham số:
        df (pd.DataFrame): DataFrame chứa các dự đoán với các cột (bao gồm drug1, drug2, perpetrator, Severity, ...).

    Trả về:
        list: Danh sách các chỉ số (index) của các dòng dự đoán được xác định là xung đột.
    """
    reported_double = {}
    reported_case = []
    for row in df.itertuples():
        drug_pair_in_order = (row[3], row[4])
        if drug_pair_in_order not in reported_double:
            reported_double[drug_pair_in_order] = (row[0], row[-2])
        elif row[-2] != reported_double[drug_pair_in_order][1]:
            reported_case += [row[0], reported_double[drug_pair_in_order][0]]

    df = df.loc[reported_case]

    severity_score_dict = {'Major': 5, 'Moderate': 4, 'Minor': 3, 'Not severe': 2, 'Unknown': 1}
    conflicting_pairs = {}
    for row in df.itertuples():
        perpet = getattr(row, 'perpetrator')
        victim = getattr(row, 'victim')
        pair = (perpet, victim)
        if pair not in conflicting_pairs:
            conflicting_pairs[pair] = {}
            conflicting_pairs[pair][row[0]] = getattr(row, 'action'), severity_score_dict[getattr(row, 'Severity')]
        else:
            conflicting_pairs[pair][row[0]] = getattr(row, 'action'), severity_score_dict[getattr(row, 'Severity')]
    idx_list = []
    for drug_pair, conflicted_pred in conflicting_pairs.items():
        filtered_result = _get_unidirectional_pred(conflicted_pred)
        if len(filtered_result) > 0:
            for k, _ in filtered_result.items():
                idx_list.append(k)
    final = list(set(reported_case) - set(idx_list))
    return final
def annotated_with_severity_result(input_file, output_file):
    # Đọc file gốc
    df = pd.read_csv(input_file)
    # Hàm ánh xạ mức độ nghiêm trọng từ DDI_prob
    def map_severity(prob):
        if prob >= 0.9:
            return "Major"
        elif prob >= 0.7:
            return "Moderate"
        elif prob >= 0.5:
            return "Minor"
        elif prob >= 0.3:
            return "Not severe"
        else:
            return "Unknown"

    # Tạo cột 'Final severity'
    df['Final severity'] = df['DDI_prob'].apply(map_severity)

    # Lưu lại file mới
    df.to_csv(output_file, index=False)
