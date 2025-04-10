import streamlit as st

st.markdown("SmartRx là một ứng dụng AI (web) giúp kiểm tra tương tác thuốc - thuốc và thuốc - thực phẩm, giúp người dùng sử dụng hiệu quả, và đảm bảo sức khỏe được Team N3N chúng tôi đề xuất và phát triển trong khuôn khổ cuộc thi GDGOC Hackathon 2025")

st.image("smartrx_web/images/smartrx.jpg")

st.markdown("""
### 🛠️ Một số tiện ích hiện có
- **Truy cập tin tức nhanh chóng, đấy đủ**
- **Nhập danh sách thuốc hoặc đơn thuốc nhanh chóng**
- **Giải thích dễ hiểu về các tương tác thuốc**
- **Đánh giá rủi ro thông qua mô hình AI**
            
### 💻 Công nghệ

### 👨‍💻 Hướng dẫn sử dụng

Để bắt đầu, bạn chỉ cần truy cập các phần khác nhau của giao diện người dùng bằng menu bên trái. Bạn có thể tra cứu tin tức liên quan, kiểm tra danh sách thuốc, kiểm tra tương tác giữa các loại thuốc hoặc thực phẩm với nhau

### 📨 Phản hồi

Phản hồi của bạn vô cùng quan trọng đối với chúng tôi! Bạn có thể trực tiếp phản hồi tại [đây](https://github.com/CryAndRRich/smartRx) hoặc gửi mail thông qua phần **Liên hệ**

### 👨‍🏫 Chân thành cảm ơn
""", unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 