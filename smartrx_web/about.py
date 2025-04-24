import streamlit as st

st.markdown(
    "SmartRx is an AI-powered web application for checking drug–drug and drug–food interactions, "
    "helping users optimize their medication regimens and safeguard their health. Developed by Team N3N "
    "as part of the GDGOC Hackathon 2025."
)

st.image("smartrx_web/images/smartrx.jpg")

st.markdown("""
### 🛠️ Key Features
- **Instant access to comprehensive news**
- **Quick entry of drug, food, or prescription lists**
- **Clear explanations of identified interactions**
- **Risk assessment powered by AI models**

### 💻 Technology Stack
SmartRx builds upon the work of [JY Ryu et al., 2018](https://www.pnas.org/doi/10.1073/pnas.1803294115?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed) 
and has been further enhanced to integrate continuously updated datasets.

We leverage modern Google Cloud services for scalable model deployment and secure, flexible storage of user data.

### 👨‍💻 How to Use
Navigate through the left-hand menu to access different modules. You can:
- Browse the latest relevant news
- Input lists of drugs, foods, or full prescriptions
- Check interactions between any two items
- Review clear, contextual interaction explanations

### 📨 Feedback
Your feedback is invaluable! Share your thoughts on our [GitHub repository](https://github.com/CryAndRRich/smartRx) or send us an email via the **Contact** section.

### 👨‍🏫 Acknowledgments
- GDGOC Hackathon Vietnam 2025 organizers for hosting this event
- Our mentor [Chiem Tri Quang](https://vn.linkedin.com/in/ctquang89?trk=public_profile_browsemap_profile-result-card_result-card_full-click) for guidance and support
""", unsafe_allow_html=True)

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
