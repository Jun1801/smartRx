import streamlit as st
import json
import os

from smartrx_web.utils import *

# Ensure the food-check JSON exists
if not os.path.exists(json_food_check_path):
    with open(json_food_check_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

st.markdown(
    "Want to eat something but worried it might interfere with your medication? "
    "**SmartRx** will help you check interactions between your chosen foods and the drugs you're taking!"
)

food_name = st.text_input("Enter a food name:", "e.g.: Orange")

if st.button("Add Food"):
    if food_name:
        name, food_info = get_food_info(food_name)
        food_name = name.strip()
        if food_info:
            with open(json_food_check_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data[food_name] = food_info

            with open(json_food_check_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            st.success(f"Successfully added {food_name}.")
        else:
            st.error("Unable to retrieve data from the API.")
    else:
        st.error("Please enter a food name.")

# Load current food checks
with open(json_food_check_path, "r", encoding="utf-8") as f:
    foods_data = json.load(f)

if foods_data:
    for food_name, info in foods_data.items():
        with st.expander(food_name):
            st.markdown(f"**Food Name:** {info.get('food_name', food_name)}")
            st.markdown(f"**Food Information:** {info.get('food_information', 'Not available')}")

            if st.button("Delete", key=f"delete_{food_name}"):
                foods_data.pop(food_name)
                with open(json_food_check_path, "w", encoding="utf-8") as f:
                    json.dump(foods_data, f, ensure_ascii=False, indent=4)
                st.rerun()
else:
    st.info("No food items have been added yet.")

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
