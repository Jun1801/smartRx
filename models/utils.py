from typing import Dict, List, Tuple

import pandas as pd
import tqdm
import re
import csv

pd.set_option("mode.chained_assignment", None)

def read_information_file(information_file: str) -> Dict[str, str]:
    """
    Parse a file containing interaction types and template sentences into a mapping.

    Parameters:
        information_file: Path to a tab-delimited file where the first line is a header,
            and each subsequent line has the format:
                "<interaction_type>\t<sentence>"
            The interaction_type token begins with a symbol (e.g., "#") which will be removed.

    Returns:
        Dict[str, str]: A dictionary mapping interaction_type (without leading symbol) to its template sentence.
    """
    interaction_info: Dict[str, str] = {}
    with open(information_file, "r") as fp:
        fp.readline()  # Skip header
        for line in fp:
            parts = line.strip().split("\t")
            key = parts[0].strip()[1:]
            sentence = parts[1].strip()
            interaction_info[key] = sentence
    return interaction_info


def read_drug_information(drug_information_file: str) -> Dict[str, List[str]]:
    """
    Load drug-target associations from a DrugBank tab-delimited file.

    Parameters:
        drug_information_file: Path to a file where each line contains multiple fields,
            including:
            - drugbank_id (column 0)
            - target (column 5)
            - action (column 7)

    Returns:
        Dict[str, List[str]]: A mapping from drugbank_id to a list of associated targets,
            including only entries where both action and target are not "None".
    """
    drug_info: Dict[str, List[str]] = {}
    with open(drug_information_file, "r") as fp:
        for line in fp:
            fields = line.strip().split("\t")
            drug_id = fields[0].strip()
            target = fields[5].strip()
            action = fields[7].strip()
            if action != "None" and target != "None":
                drug_info.setdefault(drug_id, []).append(target)
    return drug_info


def read_drug_enzyme_information(drug_enzyme_information_file: str) -> Dict[str, List[str]]:
    """
    Load drug-enzyme (uniprot) associations from a tab-delimited file.

    Parameters:
        drug_enzyme_information_file: Path to a file where each line includes:
            - drugbank_id (column 0)
            - uniprot_id (column 4)
            - action (column 5)

    Returns:
        Dict[str, List[str]]: A mapping from drugbank_id to a list of associated uniprot_ids,
            including only entries where both uniprot_id and action are not "None".
    """
    enzyme_info: Dict[str, List[str]] = {}
    with open(drug_enzyme_information_file, "r") as fp:
        for line in fp:
            fields = line.strip().split("\t")
            drug_id = fields[0].strip()
            uniprot_id = fields[4].strip()
            action = fields[5].strip()
            if uniprot_id != "None" and action != "None":
                enzyme_info.setdefault(drug_id, []).append(uniprot_id)
    return enzyme_info


def read_known_DDI_information(known_DDI_file: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Read known drug-drug interactions (DDI) and split left/right drug lists per interaction type.

    Parameters:
        known_DDI_file: Path to a tab-delimited file where the first line is a header,
            and each subsequent line has the format:
                "<left_drug>\t<right_drug>\t<interaction_type>"

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
            - left_ddi_info: mapping from interaction_type to unique list of left drugs.
            - right_ddi_info: mapping from interaction_type to unique list of right drugs.
    """
    left_ddi: Dict[str, List[str]] = {}
    right_ddi: Dict[str, List[str]] = {}
    with open(known_DDI_file, "r") as fp:
        fp.readline()  # Skip header
        for line in fp:
            left_drug, right_drug, interaction_type = [x.strip() for x in line.strip().split("\t")]
            left_ddi.setdefault(interaction_type, []).append(left_drug)
            right_ddi.setdefault(interaction_type, []).append(right_drug)

    # Remove duplicates
    for key in left_ddi:
        left_ddi[key] = list(set(left_ddi[key]))
    for key in right_ddi:
        right_ddi[key] = list(set(right_ddi[key]))

    return left_ddi, right_ddi


def read_similarity_file(similarity_file: str) -> pd.DataFrame:
    """
    Load a CSV similarity matrix into a pandas DataFrame.

    Parameters:
        similarity_file: Path to a CSV file where the first column is used as the index.

    Returns:
        pd.DataFrame: DataFrame of similarity values, indexed by row drug identifiers.
    """
    return pd.read_csv(similarity_file, index_col=0)


def read_side_effect_info(df: pd.DataFrame, 
                          frequency: int = 10) -> Dict[str, str]:
    """
    Aggregate frequent side effects per drug from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns ["Drug name", "SIDE EFFECT", "MEAN"].
        frequency (int): Minimum mean frequency threshold to include a side effect (default: 10).

    Returns:
        Dict[str, str]: Mapping from lowercase drug name to a semicolon-separated string of
            "SIDE_EFFECT(XX.X%)" entries for effects meeting the threshold, or an empty string if none.
    """
    filtered = df[df["MEAN"] >= frequency]
    result: Dict[str, str] = {}
    for drug, subdf in filtered.groupby("Drug name"):
        effects = [f"{row["SIDE EFFECT"]}({row["MEAN"]:.1f}%)" for _, row in subdf.iterrows()]
        result[drug.lower()] = ";".join(effects)
    return result


def DDI_result_supplement(input_file: str, 
                          output_file: str, 
                          interaction_info_file: str) -> None:
    """
    Supplement raw DDI predictions with human-readable sentences.

    Reads:
      - input_file: CSV of prediction results with columns ["Drug pair", "Predicted class", "Score", "STD"]
      - interaction_info_file: CSV mapping numeric DDI types to template sentences (columns "type","sentence")

    Generates output_file with additional columns:
      ["Prescription","Drug_pair","DDI_type","Sentence","Predicted class","Score","STD"]

    Parameters:
        input_file: Path to raw prediction CSV.
        output_file: Path to write supplemented CSV.
        interaction_info_file: Path to CSV mapping DDI types to sentences.
    """
    predict_df = pd.read_csv(input_file)
    interaction_df = pd.read_csv(interaction_info_file)

    predict_df["Prescription"] = predict_df["Drug pair"].apply(lambda x: int(x.split("_")[0]))
    predict_df["Drug_pair"] = predict_df["Drug pair"]
    predict_df["DDI_type"] = predict_df["Predicted class"].astype(str)

    interaction_df["type"] = interaction_df["type"].astype(int)
    sentence_map = interaction_df.set_index("type")["sentence"]
    predict_df["Sentence"] = predict_df["DDI_type"].map(sentence_map)

    cols = ["Prescription","Drug_pair","DDI_type","Sentence","Predicted class","Score","STD"]
    predict_df.to_csv(output_file, columns=cols, index=False)


def annotate_DDI_results(DDI_output_file: str,
                         similarity_file: str,
                         known_DDI_file: str,
                         output_file: str,
                         side_effect_information_file: str,
                         model_threshold: float,
                         structure_threshold: float) -> None:
    """
    Annotate DDI predictions with side effects and structurally similar approved drugs.

    Steps:
      1. Load known DDI lists, similarity matrix, raw predictions, and side effect data.
      2. For each prediction, determine confidence and filter by thresholds.
      3. List approved drugs with similarity >= structure_threshold.
      4. Write annotated results with columns:
         ["Prescription","Drug_pair","Interaction_type","Sentence","DDI_prob","DDI_prob_std",
          "Confidence_DDI","Side effects (left)","Side effects (right)",
          "Similar approved drugs (left)","Similar approved drugs (right)","drug1","drug2"]

    Parameters:
        DDI_output_file: CSV from DDI_result_supplement.
        similarity_file: CSV similarity matrix path.
        known_DDI_file: Known DDI mapping path.
        output_file: Path to write annotated CSV.
        side_effect_information_file: Tab-delimited file of side effect data.
        model_threshold: Minimum adjusted score for confidence.
        structure_threshold: Tanimoto threshold for structural similarity.
    """
    # Load reference data
    left_ddi_info, right_ddi_info = read_known_DDI_information(known_DDI_file)
    similarity_df = read_similarity_file(similarity_file)
    predictions = pd.read_csv(DDI_output_file)
    side_effect_df = pd.read_csv(side_effect_information_file, sep='\t')
    drug_side_effect_info = read_side_effect_info(side_effect_df, frequency=10)

    with open(output_file, 'w', newline='') as out_fp:
        writer = csv.writer(out_fp)
        writer.writerow([
            'Prescription', 'Drug_pair', 'Interaction_type', 'Sentence',
            'DDI_prob', 'DDI_prob_std', 'Confidence_DDI',
            'Side effects (left)', 'Side effects (right)',
            'Similar approved drugs (left)', 'Similar approved drugs (right)',
            'drug1', 'drug2'
        ])

        for row in tqdm.tqdm(predictions.itertuples(index=False), total=len(predictions)):
            rec = row._asdict()
            prescription = str(rec['Prescription'])
            drug_pair = rec['Drug_pair']
            interaction_type = str(rec['DDI_type'])
            sentence = rec['Sentence']
            score = float(rec['Score'])
            std = float(rec['STD'])
            confidence = int(score - std/2 > model_threshold)

            # Extract left and right drug labels
            m = re.match(r'\d+_(.+)', drug_pair)
            pair = m.group(1) if m else drug_pair
            m2 = re.match(r'(.+\([^()]+\))_(.+\([^()]+\))', pair)
            if not m2:
                print(f"[!] Drug_pair format error: {drug_pair}")
                continue
            left_label, right_label = m2.groups()

            # Extract identifier inside parentheses
            left_id = re.findall(r'\(([^()]+)\)$', left_label)
            right_id = re.findall(r'\(([^()]+)\)$', right_label)
            if not left_id or not right_id:
                print(f"[!] Could not parse identifiers from: {drug_pair}")
                continue

            left_id, right_id = left_id[0], right_id[0]
            left_se = drug_side_effect_info.get(left_id, '')
            right_se = drug_side_effect_info.get(right_id, '')

            # Select structurally similar approved drugs
            left_sim = []
            right_sim = []

            approved_left = left_ddi_info.get(interaction_type, [])
            if left_label in similarity_df.index and approved_left:
                sims = similarity_df.loc[left_label, approved_left]
                left_sim = list(sims[sims >= structure_threshold].index)

            approved_right = right_ddi_info.get(interaction_type, [])
            if right_label in similarity_df.index and approved_right:
                sims = similarity_df.loc[right_label, approved_right]
                right_sim = list(sims[sims >= structure_threshold].index)

            writer.writerow([
                prescription, drug_pair, interaction_type, sentence,
                score, std, confidence,
                left_se, right_se,
                ';'.join(left_sim), ';'.join(right_sim),
                left_label, right_label
            ])


def map_severity(prob: float) -> str:
    """
    Map a probability score to a severity category.

    Parameters:
        prob: Probability score between 0 and 1.

    Returns:
        str: One of ["Major","Moderate","Minor","Not severe","Unknown"].
    """
    if prob >= 0.9:
        return "Major"
    if prob >= 0.7:
        return "Moderate"
    if prob >= 0.5:
        return "Minor"
    if prob >= 0.3:
        return "Not severe"
    return "Unknown"


def summarize_prediction_outcome(result_file: str,
                                 output_file: str,
                                 information_file: str) -> None:
    """
    Generate natural-language summaries of DDI predictions by filling templates.

    Process:
      - Load sentence templates via read_information_file.
      - Read final prediction CSV (with interaction_type, score, STD, etc.).
      - Extract drug1 and drug2 from "Drug_pair".
      - Replace placeholders "#Drug1" and "#Drug2" in templates.
      - Determine final severity label via map_severity.
      - Write CSV with columns:
        ["Prescription","Drug_pair","DDI_type","Sentence","Final severity","Score","STD",
         "Side_effects (left)","Side_effects (right)"]

    Parameters:
        result_file: Path to prediction CSV.
        output_file: Path to write summary CSV.
        information_file: Path to interaction templates file.
    """
    templates = read_information_file(information_file)
    with open(result_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        with open(output_file, "w", newline="") as out_fp:
            writer = csv.writer(out_fp)
            writer.writerow(["Prescription","Drug_pair","DDI_type","Sentence",
                             "Final severity","Score","STD",
                             "Side_effects (left)","Side_effects (right)"])
            for row in reader:
                presc = row["Prescription"].strip()
                raw_pair = row["Drug_pair"].strip()
                dtype = row["Interaction_type"].strip()
                side_l = row.get("Side effects (left)","").strip()
                side_r = row.get("Side effects (right)","").strip()
                score = float(row.get("DDI_prob", row.get("Score", 0)))
                std = float(row.get("DDI_prob_std", row.get("STD", 0)))

                # Remove numeric prefix if present
                m = re.match(r"\d+_(.+)", raw_pair)
                pair = m.group(1) if m else raw_pair
                m2 = re.match(r"(.+\([^()]+\))_(.+\([^()]+\))", pair)
                if not m2:
                    continue
                d1_full, d2_full = m2.groups()
                d1 = d1_full.split("(")[0]
                d2 = d2_full.split("(")[0]

                template = templates.get(dtype, "")
                sentence = template.replace("#Drug1", d1).replace("#Drug2", d2)
                severity = map_severity(score)

                writer.writerow([presc, f"{d1}_{d2}", dtype, sentence,
                                 severity, score, std, side_l, side_r])


def processing_network(df: pd.DataFrame,
                       type_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate pharmacokinetic DDI rows with action, perpetrator, and victim labels.

    Parameters:
        df (pd.DataFrame): DataFrame of DDI predictions including columns ["Interaction_type","drug1","drug2"].
        type_df (pd.DataFrame): Mapping of interaction types to "action" and "perpetrator" columns.

    Returns:
        pd.DataFrame: Subset DataFrame of pharmacokinetic interactions with added columns ["action","perpetrator","victim"].
    """
    pk_types = list(type_df["type"])
    action_map = dict(zip(type_df["type"], type_df["action"]))
    perp_map = dict(zip(type_df["type"], type_df["perpetrator"]))

    subset = df[df["Interaction_type"].isin(pk_types)].copy()
    subset["action"] = subset["Interaction_type"].map(action_map)
    subset["perpetrator"] = subset.apply(
        lambda row: row["drug2"] if perp_map[row["Interaction_type"]] == "#Drug2" else row["drug1"], axis=1)
    subset["victim"] = subset.apply(
        lambda row: row["drug1"] if perp_map[row["Interaction_type"]] == "#Drug2" else row["drug2"], axis=1)
    return subset


def _get_unidirectional_pred(tmp: Dict[int, Tuple[str, int]]) -> Dict[int, Tuple[str, int]]:
    """
    From conflicting bidirectional predictions, select a single predominant direction.

    Parameters:
        tmp (Dict[int, Tuple[str,int]]): Mapping of row index to (direction_label, score).

    Returns:
        Dict[int, Tuple[str,int]]: Subset with entries only in the direction having the highest score,
            or empty if ambiguity remains.
    """
    best_dir = None
    best_score = -float("inf")
    best_key = None
    for key, (direction, score) in tmp.items():
        if score > best_score:
            best_score = score
            best_dir = direction
            best_key = key
        elif score == best_score and direction != best_dir:
            best_key = None
    if best_key is None:
        return {}
    return {k: v for k, v in tmp.items() if v[0] == best_dir}


def find_conflicts(df: pd.DataFrame) -> List[int]:
    """
    Identify conflicting DDI predictions between reversed drug pairs.

    Steps:
      - Find pairs reported in both orders with different actions.
      - Use _get_unidirectional_pred to resolve a single consistent direction.
      - Return indices of rows flagged as conflicts after filtering.

    Parameters:
        df (pd.DataFrame): DataFrame of annotated DDI results with columns ["perpetrator","victim","Severity"].

    Returns:
        List[int]: List of DataFrame indices corresponding to unresolved conflicts.
    """
    # Collect reported pairs and detect mismatches
    reported: Dict[Tuple[str,str], Tuple[int,str]] = {}
    mismatch_indices: List[int] = []
    for row in df.itertuples():
        pair = (row.perpetrator, row.victim)
        idx = row.Index
        action = row.action
        if pair not in reported:
            reported[pair] = (idx, action)
        elif reported[pair][1] != action:
            mismatch_indices.extend([reported[pair][0], idx])

    subset = df.loc[mismatch_indices]
    severity_map = {"Major":5,"Moderate":4,"Minor":3,"Not severe":2,"Unknown":1}
    conflicts: Dict[Tuple[str,str], Dict[int, Tuple[str,int]]] = {}
    for row in subset.itertuples():
        pair = (row.perpetrator, row.victim)
        entry = (row.action, severity_map[row.Severity])
        conflicts.setdefault(pair, {})[row.Index] = entry

    resolved_indices: List[int] = []
    for pair, tmp in conflicts.items():
        keep = _get_unidirectional_pred(tmp)
        resolved_indices.extend(keep.keys())

    # Return those still in report but not resolved
    return list(set(mismatch_indices) - set(resolved_indices))


def annotated_with_severity_result(input_file: str, 
                                   output_file: str) -> None:
    """
    Append a "Final severity" column to an existing DDI result CSV.

    Parameters:
        input_file: Path to CSV with "DDI_prob" column.
        output_file: Path to write the updated CSV.
    """
    df = pd.read_csv(input_file)
    df["Final severity"] = df["DDI_prob"].apply(map_severity)
    df.to_csv(output_file, index=False)
