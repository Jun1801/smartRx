import streamlit as st

st.warning("#### If you’d like to report a bug 👾 or suggest a feature ⚡, please reach out below!")

contact_form = """
<form action="https://formsubmit.co/trannamhai.5d@gmail.com" method="POST" enctype="multipart/form-data">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="text" name="_subject" placeholder="Subject">
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <input type="file" class="img_btn" name="Upload Image" accept="image/png, image/jpeg">
     <br>
     <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "#### ⭐ If you’re enjoying the website experience, please give us a star on "
    "[GitHub](https://github.com/CryAndRRich/smartRx)!",
    unsafe_allow_html=True
)

# Hide Streamlit’s default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom styles
local_css("smartrx_web/style/style.css")

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
