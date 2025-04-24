import streamlit as st
import json
import os

from smartrx_web.utils import *

# Ensure the saved-drug JSON file exists
if not os.path.exists(json_drug_list_path):
    with open(json_drug_list_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

# Input for adding a new medication
drug_name = st.text_input("Enter medication name:", "e.g.: Ibuprofen")

if st.button("Add to Saved List"):
    if drug_name:
        name, drug_info = get_drug_info(drug_name)
        drug_name = name.strip()
        if drug_info:
            with open(json_drug_list_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data[drug_name] = drug_info
            with open(json_drug_list_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            st.success(f"Successfully added {drug_name} to your saved list.")
        else:
            st.error("Unable to retrieve data from the API.")
    else:
        st.error("Please enter a medication name.")

# Load and display saved medications
with open(json_drug_list_path, "r", encoding="utf-8") as f:
    drugs_data = json.load(f)

if drugs_data:
    for drug_name, info in drugs_data.items():
        with st.expander(drug_name):
            st.markdown(f"**Medication Name:** {info.get('drug_name', drug_name)}")
            st.markdown(f"**Intended Use:** {info.get('purpose', 'Not available')}")
            st.markdown(f"**Usage Instructions:** {info.get('general_usage', 'Not available')}")
            if st.button("Remove from Saved List", key=f"delete_{drug_name}"):
                # Remove the medication and update the JSON
                drugs_data.pop(drug_name)
                with open(json_drug_list_path, "w", encoding="utf-8") as f:
                    json.dump(drugs_data, f, ensure_ascii=False, indent=4)
                st.rerun()
else:
    st.info("No medications have been saved yet.")
