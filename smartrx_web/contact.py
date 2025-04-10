import streamlit as st

st.warning("#### Nếu bạn muốn Báo cáo lỗi 👾 hoặc Đề xuất tính năng ⚡!")

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

st.markdown("#### ⭐ Nếu bạn hài lòng với trải nghiệm trên website, xin hãy để lại một ngôi sao tại [đây](https://github.com/CryAndRRich/smartRx)", unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("smartrx_web/style/style.css")

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 