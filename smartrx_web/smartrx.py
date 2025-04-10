import streamlit as st

st.markdown("### 👨‍⚕️ SmartRx - AI for checking drugs and food interactions ")

st.image("smartrx_web/images/smartrx.jpg")

st.info("Original Repository on [Github](https://github.com/CryAndRRich/smartRx)")

st.markdown("---")

st.markdown("### 🌎 Thực trạng hiện nay")

with st.expander("Thế giới"):
    st.markdown(
        """
        Theo các nghiên cứu y học và báo cáo của các tổ chức sức khỏe toàn cầu, **polypharmacy** – việc sử dụng đồng thời nhiều loại thuốc, đặc biệt là **năm** thuốc trở lên – được xác định là một trong những yếu tố **rủi ro quan trọng** ở người cao tuổi. Cụ thể:

        - Có tới gần **500 triệu người** cao tuổi trên toàn cầu đang đối mặt với nguy cơ polypharmacy một cách đáng kể. Con số này cho thấy một tỷ lệ đáng kể người cao tuổi có nguy cơ bị **tác dụng phụ** và **biến chứng** do việc dùng nhiều loại thuốc.
        - Khi sử dụng từ **năm loại thuốc trở lên**, nguy cơ xảy ra các phản ứng thuốc bất lợi có thể tăng đến **khoảng 50%** so với những người dùng ít thuốc hơn. Sự gia tăng này phản ánh mức độ phức tạp trong dược lý của cơ thể người cao tuổi, khi mà khả năng chuyển hóa thuốc và cân bằng nội tiết có thể đã bị suy giảm theo tuổi tác.
        """)

with st.expander("Việt Nam"):
    st.markdown(
        """
        Một [bài nghiên cứu](https://jppres.com/jppres/outpatient-prescription-drug-interactions-in-vietnam/) của một nhóm sinh viên Cần Thơ tại một khoa ngoại trú của một trung tâm y tế liên kết với thành phố từ tháng 1 đến tháng 6 năm 2023 chỉ ra rằng có đến **36,7%** đơn thuốc chứa **tương tác thuốc bất lợi**.
        """)
    
st.markdown("""
### 🎯 Mục tiêu
**SmartRx** là một web app, giúp kiểm tra tương tác thuốc bằng cách đọc ảnh mã vạch của thuốc và hiển thị trang chi tiết thuốc, bao gồm thông tin mô tả, tác dụng phụ, cũng như các tương tác giữa thuốc này và các loại thuốc có trong danh sách của người dùng. Chức năng kiểm tra tương tác với thực phẩm cho phép người dùng nhập tên hoặc quét hình ảnh các loại thực phẩm và sau đó cảnh báo lại nếu có bất kỳ tương tác nào với các loại thuốc trong danh sách. 
""")

st.image("smartrx_web/images/usecase.png")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 