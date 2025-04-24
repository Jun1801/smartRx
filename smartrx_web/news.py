import requests
import streamlit as st
import tomllib

st.markdown(
    "Easily search and look up information on any medications or foods you need—"
    "just enter a keyword and let SmartRx handle the rest!"
)

# Load API credentials from TOML config
with open(".streamlit/api_config.toml", "rb") as f:
    config = tomllib.load(f)

api_key = config["google"]["google_search_api"]
cx = config["google"]["cx"]

# Search input
query = st.text_input("Enter search keyword:", "drug–drug interaction")

if query:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "sort": "date"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json().get("items", [])
        if results:
            for item in results:
                st.subheader(item.get("title"))
                st.write(item.get("snippet"))
                st.markdown(f"[Read more]({item.get('link')})")
                st.markdown("---")
        else:
            st.warning("No results found.")
    else:
        st.error(f"API request error: {response.status_code} - {response.text}")
