import streamlit as st

st.markdown("### 👨‍⚕️ SmartRx – AI for Checking Drug–Drug and Drug–Food Interactions")
st.markdown(
    "Welcome to SmartRx, an AI-powered tool for detecting drug–drug and drug–food interactions "
    "to help you manage your medications with confidence."
)
st.image("smartrx_web/images/smartrx.jpg")

st.info("Original repository on [GitHub](https://github.com/CryAndRRich/smartRx)")

st.markdown("---")

st.markdown("### 🌎 Current Challenges")
st.markdown("""
Patients today often have multiple conditions and visit doctors frequently, which can lead to forgetting 
which medications to take and when—and whether combining different drugs might pose risks.

At the same time, varied diets and lifestyles make it difficult to follow strict medical advice, increasing 
the likelihood of adverse drug–food interactions.
""")

with st.expander("Global"):
    st.markdown("""
    According to medical research and reports from global health organizations, **polypharmacy**—the concurrent 
    use of multiple medications, especially **five or more**—is identified as a significant **risk factor** among 
    older adults. Specifically:

    - Nearly **500 million older adults** worldwide face substantial polypharmacy risks, highlighting the prevalence 
      of potential **side effects** and **complications** from multiple medications.
    - When taking **five or more medications**, the risk of adverse drug reactions can increase by **around 50%** 
      compared to those on fewer drugs, reflecting the complexities of aging pharmacology and reduced metabolic capacity.
    """)

with st.expander("Vietnam"):
    st.markdown("""
    A [study](https://jppres.com/jppres/outpatient-prescription-drug-interactions-in-vietnam/) conducted 
    at an outpatient clinic in Can Tho from January to June 2023 found that **36.7%** of prescriptions—over 
    one-third—contained **potentially harmful drug interactions**.
    """)

st.markdown("""
### 🎯 Objectives
**SmartRx** enables patients to store complete medication and diagnosis histories, detect adverse interactions, 
receive reminders, and easily check drug–drug or drug–food interactions.

Acting as an electronic health record for any user, SmartRx offers a simple, effective way to manage prescriptions.
""")

# Hide Streamlit’s default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
