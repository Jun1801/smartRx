from typing import Dict, List, Tuple

import pandas as pd
import tqdm
import re

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
        interaction_type = line_split[0].strip()
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


def get_side_effects(df: pd.DataFrame, 
                     target_drug: str, 
                     frequency: int = 10) -> str:
    """
    Lấy thông tin side effects của một thuốc dựa trên DataFrame chứa thông tin side effects.
    
    Tham số:
        df: DataFrame chứa thông tin side effects với các cột bao gồm 'Drug name', 'SIDE EFFECT', 'MEAN'.
        target_drug: Tên thuốc cần tra cứu side effects.
        frequency: Ngưỡng giá trị trung bình (MEAN) để chọn lọc side effect. Mặc định là 10.
        
    Trả về:
        str: Chuỗi được ghép các side effect của thuốc, mỗi side effect có định dạng "SIDE_EFFECT(XX.X%)",
             được phân cách bởi dấu chấm phẩy. Nếu không có side effects nào thỏa mãn điều kiện, trả về chuỗi rỗng.
    """
    new_df = df[df['Drug name'] == target_drug]
    new_df = new_df[new_df['MEAN'] >= frequency]
    drug_side_effect_info = {}
    
    for each_drug, each_df in new_df.groupby('Drug name'):
        string_list = []
        for each_index, each_df in each_df.iterrows():
            side_effect = each_df['SIDE EFFECT']
            mean_frequency = each_df['MEAN']
            string_list.append('%s(%.1f%%)' % (side_effect, mean_frequency))
            
        drug_side_effect_info[each_drug] = ';'.join(string_list)
    
    string_list = []
    for each_drug in drug_side_effect_info:
        string_list.append('%s' % (drug_side_effect_info[each_drug]))
        
    side_effect_string = ';'.join(string_list)
    return side_effect_string


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


def annotate_DDI_results(DDI_output_file: str, 
                         similarity_file: str, 
                         known_DDI_file: str, 
                         output_file: str, 
                         side_effect_information_file: str, 
                         model_threshold: float, 
                         structure_threshold: float) -> None:
    """
    Ghi annotation cho kết quả dự đoán DDI và xuất ra file kết quả.
    
    Quy trình:
        - Đọc thông tin thuốc, enzyme, DDI đã biết, bảng tương đồng, và thông tin side effect.
        - Duyệt qua từng dòng dự đoán trong file DDI_output_file.
        - Lấy thông tin side effect cho từng thuốc dựa trên các giá trị trong file side effect.
        - Tính toán Confidence_DDI dựa trên model_threshold.
        - Lấy các thuốc tương đồng với cấu trúc đạt yêu cầu dựa trên structure_threshold từ bảng similarity.
        - Ghi toàn bộ thông tin annotation ra file output_file.
    
    Tham số:
        DDI_output_file: Đường dẫn tới file chứa kết quả dự đoán DDI (dạng TSV).
        similarity_file: Đường dẫn tới file CSV chứa bảng tương đồng.
        known_DDI_file: Đường dẫn tới file thông tin DDI đã biết.
        output_file: Đường dẫn file sẽ ghi kết quả annotation.
        side_effect_information_file: Đường dẫn tới file side effect (dạng TSV).
        model_threshold: Ngưỡng để xác định Confidence_DDI.
        structure_threshold: Ngưỡng để chọn lọc cấu trúc tương đồng.
    """
    
    left_ddi_info, right_ddi_info = read_known_DDI_information(known_DDI_file)    
    similarity_df = read_similarity_file(similarity_file)
    DDI_prediction_df = pd.read_csv(DDI_output_file, sep='\t')
    side_effect_df = pd.read_csv(side_effect_information_file, sep='\t')
    drug_side_effect_info = read_side_effect_info(side_effect_df, frequency=10)
    fp = open(output_file, 'w')
    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %
             ('Prescription', 'Drug_pair', 'Interaction_type', 'Sentence', 'DDI_prob', 'DDI_prob_std', 'Confidence_DDI',   
              'Side effects (left)', 'Side effects (right)', 'Similar approved drugs (left)', 'Similar approved drugs (right)',
              'drug1', 'drug2'))

    for row in tqdm.tqdm(DDI_prediction_df.itertuples(), total=len(DDI_prediction_df)):
        prescription = str(getattr(row, 'Prescription'))
        drug_pair = getattr(row, 'Drug_pair')
        
        left_drug, right_drug = drug_pair.split('_')
        DDI_type = str(getattr(row, 'DDI_type'))
        sentence = getattr(row, 'Sentence')
        score = getattr(row, 'Score')
        std = getattr(row, 'STD')
        Confidence_DDI = 0
        left_drug_side_effect = ''
        right_drug_side_effect = ''

        # Lấy thông tin side effect của từng thuốc từ chuỗi tên (có chứa dấu ngoặc)
        left_comp, right_comp = re.findall('.*\((.*)\)$', left_drug)[0], re.findall('.*\((.*)\)$', right_drug)[0]
        if left_comp in drug_side_effect_info:
            left_drug_side_effect = drug_side_effect_info[left_comp]
        if right_comp in drug_side_effect_info:
            right_drug_side_effect = drug_side_effect_info[right_comp]

        if score - std/2 > model_threshold:
            Confidence_DDI = 1
            
        left_corresponding_drugs = left_ddi_info[DDI_type]
        right_corresponding_drugs = right_ddi_info[DDI_type]
        
        left_drug_similarity_df = similarity_df.loc[left_drug][left_corresponding_drugs]
        left_selected_drugs = list(left_drug_similarity_df[left_drug_similarity_df >= structure_threshold].index)
        
        right_drug_similarity_df = similarity_df.loc[right_drug][right_corresponding_drugs]
        right_selected_drugs = list(right_drug_similarity_df[right_drug_similarity_df >= structure_threshold].index)

        left_drug_annotation_string = ';'.join(left_selected_drugs)
        right_drug_annotation_string = ';'.join(right_selected_drugs)
        fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %
                 (prescription, drug_pair, DDI_type, sentence, score, std, Confidence_DDI,
                  left_drug_side_effect, right_drug_side_effect, left_drug_annotation_string, 
                  right_drug_annotation_string, left_drug, right_drug))
    
    fp.close()        
    return


def summarize_prediction_outcome(result_file: str, 
                                 output_file: str, 
                                 information_file: str) -> None:
    """
    Tóm tắt kết quả dự đoán bằng cách thay thế mẫu câu với tên các thuốc.
    
    Quy trình:
        - Đọc file thông tin sentence template dựa trên DDI_class từ file information_file.
        - Đọc file kết quả dự đoán result_file (dạng TSV), bỏ qua header.
        - Với mỗi dòng, lấy thông tin gồm: Prescription, Drug_pair, DDI_class, predicted_score, predicted_std.
        - Tách thông tin trong Drug_pair để lấy tên các thuốc (drug1, drug2) và thay thế vào template sentence.
        - Ghi kết quả ra output_file với các cột: Prescription, Drug_pair, DDI_class, Sentence (sau khi thay thế), Score, STD.
    
    Tham số:
        result_file: Đường dẫn tới file kết quả dự đoán gốc.
        output_file: Đường dẫn tới file sẽ ghi kết quả tóm tắt.
        information_file: Đường dẫn tới file chứa thông tin sentence template (mỗi loại DDI có một mẫu câu).
    """
    sentence_interaction_info = read_information_file(information_file)
    
    with open(result_file, 'r') as fp:
        fp.readline()  # Bỏ qua header
        out_fp = open(output_file, 'w')
        out_fp.write('%s\t%s\t%s\t%s\t%s\t%s\n' %
                     ('Prescription', 'Drug_pair', 'DDI_type', 'Sentence', 'Score', 'STD'))
        for line in fp:
            line_split = line.strip().split('\t')
            drug_pair_info = line_split[0].strip()
            drug_pair_list = drug_pair_info.split('_')
            prescription = drug_pair_list[0]
            drug1 = drug_pair_list[1]
            drug2 = drug_pair_list[2]
            drug_pair = '%s_%s' % (drug1, drug2)
            DDI_class = line_split[1].strip()
            predicted_score = line_split[2].strip()
            predicted_std = line_split[3].strip()
        
            template_sentence = sentence_interaction_info[DDI_class]
            prediction_outcome = template_sentence.replace('#Drug1', drug1)
            prediction_outcome = prediction_outcome.replace('#Drug2', drug2)
            out_fp.write('%s\t%s\t%s\t%s\t%s\t%s\n' %
                         (prescription, drug_pair, DDI_class, prediction_outcome, predicted_score, predicted_std))
        out_fp.close()


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
    Hàm tìm hướng dự đoán duy nhất dựa vào dictionary chứa các dự đoán conflict.
    
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


def filter_final_result(annotated_result_file: str, 
                        conflicting_type_file: str, 
                        output_file: str) -> None:
    """
    Lọc kết quả cuối cùng từ file annotated_result_file dựa trên một số tiêu chí:
        1. Confidence_DDI phải bằng 1.
        2. Ít nhất một trong hai trường 'Similar approved drugs (left)' hoặc 'Similar approved drugs (right)' không rỗng.
        3. Đối với các loại model2 không có hướng (đánh dấu bằng false_types), loại bỏ các trường trùng lặp.
        4. Loại bỏ các drug pair có xung đột dựa trên thông tin từ conflicting_type_file.
    
    Tham số:
        annotated_result_file: Đường dẫn tới file kết quả annotated ban đầu (dạng TSV).
        conflicting_type_file: Đường dẫn tới file chứa thông tin các loại xung đột (dạng TSV).
        output_file: Đường dẫn tới file sẽ ghi kết quả cuối cùng sau khi lọc.
    """
    df = pd.read_csv(annotated_result_file, sep='\t')
    # Filter 1: Chỉ giữ các dòng có Confidence_DDI == 1 
    df1 = df[df['Confidence_DDI'] == 1]
    # Filter 2: Ít nhất một trường "Similar approved drugs" không rỗng
    df1 = df1[(df1['Similar approved drugs (left)'].isna() == False) | (df1['Similar approved drugs (right)'].isna() == False)]
    
    # Chỉ giữ các cột cần thiết
    df1 = df1[['drug1', 'drug2', 'Interaction_type', 'Sentence', 'Final severity', 'Side effects (left)', 'Side effects (right)']]
    df1.rename(columns={'Final severity': 'Severity'}, inplace=True)
    
    # Filter 4: Loại bỏ duplicates cho các loại model2 không có hướng (#drug1-#drug2)
    false_types = [117, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 143, 145, 146, 147,
                   148, 151, 153, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                   171, 172, 173, 174, 179, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    
    pairset_list = []
    for row in df1.itertuples():
        drug1 = getattr(row, 'drug1')
        drug2 = getattr(row, 'drug2')
        pairset = frozenset([drug1, drug2])
        pairset_list.append(pairset)
    df1['pairset'] = pairset_list
    df_sub1 = df1[df1['Interaction_type'].isin(false_types)]
    df_sub1.drop_duplicates(subset=['Interaction_type', 'pairset'], inplace=True)
    df_sub2 = df1[~df1['Interaction_type'].isin(false_types)]
    df1 = pd.concat([df_sub1, df_sub2], ignore_index=True)
    
    # Filter 5: Loại bỏ các drug pair có xung đột dựa trên file conflicting_type_file
    conflicting_types = pd.read_csv(conflicting_type_file, sep='\t')
    conc_type_df = conflicting_types[conflicting_types['category'] == 'concentration']
    metab_type_df = conflicting_types[conflicting_types['category'] == 'metabolism']
    Conc = processing_network(df1, conc_type_df)
    metabolism = processing_network(df1, metab_type_df)
    Conc_reported = find_conflicts(Conc)
    met_reported = find_conflicts(metabolism)
    conflicts_total = met_reported + Conc_reported
    no_conflicts = list(set(df1.index) - set(conflicts_total))
    df_final = df1.loc[no_conflicts]
    df_final = df_final[['drug1', 'drug2', 'Interaction_type', 'Sentence', 'Severity', 'Side effects (left)', 'Side effects (right)']]
    df_final.to_csv(output_file, sep='\t', index=False)


def concatenate_results(model1_result_file: str, 
                        model2_result_file: str, 
                        output_file: str) -> None:
    """
    Nối kết quả từ hai file dự đoán (model1 và model2) và ghi ra file kết quả cuối cùng.
    
    Tham số:
        model1_result_file (str): Đường dẫn tới file kết quả của model1 (dạng TSV).
        model2_result_file (str): Đường dẫn tới file kết quả của model2 (dạng TSV).
        output_file (str): Đường dẫn tới file sẽ ghi kết quả nối sau khi loại bỏ duplicates.
    """
    df_model1 = pd.read_csv(model1_result_file, sep='\t')
    df_model2 = pd.read_csv(model2_result_file, sep='\t')
    df_concat = pd.concat([df_model1, df_model2], ignore_index=True)
    df_concat = df_concat.drop_duplicates()
    df_concat.to_csv(output_file, sep='\t', index=False)
