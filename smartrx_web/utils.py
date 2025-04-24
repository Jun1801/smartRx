import csv
from typing import Tuple, Dict, Any, List
import streamlit as st
import requests
import tomllib
import json
import os

# Load API configuration
with open(".streamlit/api_config.toml", "rb") as f:
    config = tomllib.load(f)

API_KEY = config["google"]["google_gemini_api"]
JSON_DRUG_LIST_PATH = "smartrx_web/json/drug_list_info.json"
JSON_DRUG_CHECK_PATH = "smartrx_web/json/drug_check_info.json"
JSON_FOOD_CHECK_PATH = "smartrx_web/json/food_check_info.json"

URL_API = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={API_KEY}"
)

def get_drug_info(drug_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve concise information about a medication using a language model API.

    The function will prompt the model to return exactly three numbered paragraphs:
      1. "drug name|English generic name"
      2. "purpose of use"
      3. "usage instructions"

    If the exact drug name is not found, the model will choose the closest match.

    Parameters:
        drug_name: The name of the medication to look up.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing:
            - actual_name: The standardized drug name returned by the model.
            - info: A dictionary with keys:
                'drug_name', 'purpose', and 'general_usage'.
            If the API call fails, returns ({}, {}) and displays an error in Streamlit.
    """
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Please provide the most concise overview possible for the medication named '{drug_name}'. "
        "If no exact match is found, automatically substitute the closest existing medication name without notification. "
        "Return exactly three numbered paragraphs without titles: "
        "1. (Format: 'drug name|English generic name only') "
        "2. (Purpose of use, in Vietnamese) "
        "3. (Usage instructions, in Vietnamese) "
        "Ensure clarity and accuracy; return no other text."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(URL_API, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json().get("candidates", [])
        if not data:
            st.error("API returned no valid candidates.")
            return "", {}
        full_text = data[0]["content"]["parts"][0]["text"].strip()
        # Parse three sections
        try:
            sec1, rest = full_text.split("2.", 1)
            name_part = sec1.strip()[2:].strip()
            sec2, sec3 = rest.split("3.", 1)
            purpose = sec2.strip()
            usage = sec3.strip()
        except ValueError:
            st.error("Unexpected format from API response.")
            return "", {}
        result = {"drug_name": name_part, "purpose": purpose, "general_usage": usage}
        return name_part, result
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return "", {}


def get_food_info(food_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve concise information about a food item using a language model API.

    The function will prompt the model to return exactly two numbered paragraphs:
      1. "major nutrient|English generic name of nutrient"
      2. "information about that nutrient"

    If no exact match is found, the model will choose the closest existing food name.

    Parameters:
        food_name: The name of the food item to look up.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing:
            - actual_name: The standardized food name or nutrient label.
            - info: A dictionary with keys 'food_name' and 'food_information'.
            If the API call fails, returns ({}, {}) and displays an error in Streamlit.
    """
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Please provide the most concise overview possible for the food item '{food_name}'. "
        "If no exact match is found, automatically substitute the closest food name without notification. "
        "Return exactly two numbered paragraphs without titles: "
        "1. (Format: 'nutrient name in English|English generic name of the nutrient') "
        "2. (Information about that nutrient, in Vietnamese) "
        "Ensure clarity and accuracy; return no other text."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(URL_API, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json().get("candidates", [])
        if not data:
            st.error("API returned no valid candidates.")
            return "", {}
        full_text = data[0]["content"]["parts"][0]["text"].strip()
        # Parse two sections
        try:
            part1, part2 = full_text.split("2.", 1)
            name_part = part1.strip()[2:].strip()
            info_text = part2.strip()
        except ValueError:
            st.error("Unexpected format from API response.")
            return "", {}
        result = {"food_name": name_part, "food_information": info_text}
        return name_part, result
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return "", {}


def get_drug_input() -> None:
    """
    Write combined drug lists to a text file for downstream DDI processing.

    Reads two JSON files:
      - One for drugs to check interactions.
      - One for the user's saved drug list.
    Then writes both lists as two tab-separated lines to
    'data/Dataset/Input_txt/combined_drug_lists.txt'.
    """
    with open(JSON_DRUG_CHECK_PATH, 'r', encoding='utf-8') as f:
        check_list = list(json.load(f).keys())
    with open(JSON_DRUG_LIST_PATH, 'r', encoding='utf-8') as f:
        saved_list = list(json.load(f).keys())
    os.makedirs(os.path.dirname("data/Dataset/Input_txt/combined_drug_lists.txt"), exist_ok=True)
    with open("data/Dataset/Input_txt/combined_drug_lists.txt", "w", encoding='utf-8') as f:
        f.write("\t".join(check_list) + "\n")
        f.write("\t".join(saved_list) + "\n")


def processing_result(input_file: str) -> List[List[str]]:
    """
    Process the DDI result CSV and select the stronger prediction per drug pair.

    Reads a CSV with alternating rows for each drug pair prediction,
    compares the 'Score' field, and retains the row with the higher score.

    Returns a list of records, each containing:
      [drug1, drug2, sentence, formatted_score, combined_side_effects]

    Parameters:
        input_file: Path to the DDI result CSV file.

    Returns:
        List[List[str]]: Processed results for display.
    """
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i in range(0, len(rows), 2):
        if i+1 >= len(rows):
            break
        row1, row2 = rows[i], rows[i+1]
        score1 = float(row1.get("Score", 0.0) or 0.0)
        score2 = float(row2.get("Score", 0.0) or 0.0)
        selected = row1 if score1 >= score2 else row2
        pair = selected.get("Drug_pair", "_")
        drugs = pair.split("_") if "_" in pair else [pair, ""]
        sentence = selected.get("Sentence", "")
        score_str = f"{float(selected.get('Score',0)):.4f}"
        se_left = selected.get("Side_effects (left)", "")
        se_right = selected.get("Side_effects (right)", "")
        combined_se = se_left + ("; " + se_right if se_right else "")
        results.append([drugs[0], drugs[1], sentence, score_str, combined_se])
    return results


def get_result_text(result: List[str]) -> Dict[str, str]:
    """
    Enrich a DDI result with additional explanation via language model.

    Sends a prompt including the drug names, original interaction sentence,
    and score, then expects exactly four '|'-delimited sections:
      1. "Drug1 <-> Drug2|score"
      2. Translation of the interaction into Vietnamese.
      3. Detailed explanation in Vietnamese (avoid jargon).
      4. List of at least five side effects in Vietnamese, comma-separated.

    Parameters:
        result: A record from processing_result: [drug1, drug2, sentence, score, side_effects]

    Returns:
        Dict[str, str]: Mapping with keys 'drugs','score','sentence1','sentence2','side_effect',
                        or empty dict on error.
    """
    drug1, drug2, sentence, score, _ = result
    drug1_cap = drug1.capitalize()
    drug2_cap = drug2.capitalize()
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Please provide a concise overview of the adverse interaction between {drug1_cap} and {drug2_cap}. "
        f"Original interaction: {sentence}. Expand and clarify the content. "
        "Return exactly four '|' separated sections with no line breaks: "
        f"1. '{drug1_cap} <-> {drug2_cap}|{score}' "
        "2. Translate the initial interaction into Vietnamese. "
        "3. Provide a detailed Vietnamese explanation without technical jargon. "
        "4. List at least five side effects in Vietnamese, separated by commas. "
        "Ensure clarity and accuracy; return no other text."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(URL_API, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        if not candidates:
            st.error("API returned no valid candidates.")
            return {}
        full_text = candidates[0]["content"]["parts"][0]["text"].strip()
        parts = full_text.split("|")
        if len(parts) < 4:
            return {}
        return {
            "drugs": parts[0],
            "score": parts[1],
            "sentence1": parts[2],
            "sentence2": parts[3],
            "side_effect": parts[4] if len(parts) > 4 else ""
        }
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return {}

# Ensure JSON files exist on startup
for path in [JSON_DRUG_LIST_PATH, JSON_DRUG_CHECK_PATH, JSON_FOOD_CHECK_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
