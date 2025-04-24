import streamlit as st
import json
import os

from smartrx_web.utils import *

# Ensure the drug-check JSON exists
if not os.path.exists(json_drug_check_path):
    with open(json_drug_check_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

st.markdown(
    "Enter any medications you’re concerned about, and **SmartRx** will check not only the interactions among those "
    "medications but also with any drugs you’ve already saved!"
)

drug_name_input = st.text_input("Enter medication name:", "e.g.: Ibuprofen")

if st.button("Add Medication"):
    if drug_name_input:
        name, drug_info = get_drug_info(drug_name_input)
        drug_name_input = name.strip()
        if drug_info:
            with open(json_drug_check_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data[drug_name_input] = drug_info
            with open(json_drug_check_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            st.success(f"Successfully added {drug_name_input}.")
        else:
            st.error("Unable to retrieve data from the API.")
    else:
        st.error("Please enter a medication name.")

# Load current and saved drug lists
with open(json_drug_check_path, "r", encoding="utf-8") as f:
    drugs_data = json.load(f)
with open(json_drug_list_path, "r", encoding="utf-8") as f:
    drugs_list_data = json.load(f)

if drugs_data:
    for drug_name, info in drugs_data.items():
        status = "Not in saved list"
        saved_lower = [d.lower() for d in drugs_list_data.keys()]
        if drug_name.lower() in saved_lower:
            status = "Already in saved list"

        with st.expander(drug_name):
            st.markdown(f"**Medication Name:** {info.get('drug_name', drug_name)}")
            st.markdown(f"**Intended Use:** {info.get('purpose', 'Not available')}")
            st.markdown(f"**Usage Instructions:** {info.get('general_usage', 'Not available')}")
            st.markdown(f"**Status:** {status}")

            if status == "Already in saved list":
                if st.button("Delete", key=f"delete_{drug_name}"):
                    drugs_data.pop(drug_name)
                    with open(json_drug_check_path, "w", encoding="utf-8") as f:
                        json.dump(drugs_data, f, ensure_ascii=False, indent=4)
                    st.rerun()

            else:  # Not in saved list
                col1, col2, _ = st.columns([25, 70, 120])
                with col1:
                    if st.button("Delete", key=f"delete_{drug_name}"):
                        drugs_data.pop(drug_name)
                        with open(json_drug_check_path, "w", encoding="utf-8") as f:
                            json.dump(drugs_data, f, ensure_ascii=False, indent=4)
                        st.rerun()
                with col2:
                    if st.button("Add to Saved List", key=f"add_{drug_name}"):
                        drugs_list_data[drug_name] = info
                        with open(json_drug_list_path, "w", encoding="utf-8") as f:
                            json.dump(drugs_list_data, f, ensure_ascii=False, indent=4)
                        st.success(f"Successfully added {drug_name} to your saved list.")
                        st.rerun()
else:
    st.info("No medications have been added yet.")

if st.button("Check Interactions"):
    get_drug_input()

    results = processing_result(input_file="data/Result/final_result.csv")
    for res in results:
        r = get_result_text(res)
        if not r:
            continue
        with st.expander(r['drugs']):
            st.markdown(f"**Adverse Interaction:** {r['sentence1']}")
            st.markdown(f"**Interaction Details:** {r['sentence2']}")
            st.markdown(f"**Side Effects:** {r['side_effect']}")
            st.markdown(f"**Prediction Accuracy:** {r['score']}")

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
