import streamlit as st

st.markdown("SmartRx là một ứng dụng AI (web) giúp kiểm tra tương tác thuốc - thuốc và thuốc - thực phẩm, giúp người dùng sử dụng hiệu quả, và đảm bảo sức khỏe được Team N3N chúng tôi đề xuất và phát triển trong khuôn khổ cuộc thi GDGOC Hackathon 2025")

st.image("smartrx_web/images/smartrx.jpg")

st.markdown("""
### 🛠️ Một số tiện ích hiện có
- **Truy cập tin tức nhanh chóng, đấy đủ**
- **Nhập danh sách thuốc, thực phẩm hoặc đơn thuốc nhanh chóng**
- **Giải thích dễ hiểu về các tương tác thuốc**
- **Đánh giá rủi ro thông qua mô hình AI**
            
### 💻 Công nghệ
SmartRx được xây dựng dựa trên bài nghiên cứu [JY Ryu et al., 2018](https://www.pnas.org/doi/10.1073/pnas.1803294115?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed) sau đó được cải tiến để phù hợp với bộ dữ liệu được cập nhật liên tục

SmartRx sử dụng các công nghệ Google Cloud hiện đại để triển khai, mở rộng mô hình dễ dàng đồng thời lưu trữ dữ liệu người dùng bảo mật, linh hoạt

### 👨‍💻 Hướng dẫn sử dụng

Để bắt đầu, bạn chỉ cần truy cập các phần khác nhau của giao diện người dùng bằng menu bên trái. Bạn có thể tra cứu tin tức liên quan, kiểm tra danh sách thuốc, kiểm tra tương tác giữa các loại thuốc hoặc thực phẩm với nhau

### 📨 Phản hồi

Phản hồi của bạn vô cùng quan trọng đối với chúng tôi! Bạn có thể trực tiếp phản hồi tại [đây](https://github.com/CryAndRRich/smartRx) hoặc gửi mail thông qua phần **Liên hệ**

### 👨‍🏫 Chân thành cảm ơn

- Ban tổ chức GDGoC Hackathon Vietnam 2025 đã tạo sân chơi giúp chúng tôi tham gia và tạo nên ý tưởng này
- Mentor [Chiem Tri Quang](https://vn.linkedin.com/in/ctquang89?trk=public_profile_browsemap_profile-result-card_result-card_full-click) đã giúp đỡ và hỗ trợ chúng tôi hoàn thiện ý tưởng
""", unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
