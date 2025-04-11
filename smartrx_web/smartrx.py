import streamlit as st

st.markdown("### 👨‍⚕️ SmartRx - AI for checking drugs and food interactions ")
st.markdown("Chào mừng bạn đến với SmartRx, công cụ trợ giúp kiểm tra và phát hiện tương tác thuốc - thuốc hoặc thuốc - thực phẩm nhằm giúp bạn giảm đi mối lo ngại khi sử dụng thuốc")
st.image("smartrx_web/images/smartrx.jpg")

st.info("Original Repository on [Github](https://github.com/CryAndRRich/smartRx)")

st.markdown("---")

st.markdown("### 🌎 Thực trạng hiện nay")
st.markdown("""
Bệnh nhân hiện nay mắc nhiều bệnh và phải gặp bác sĩ nhiều lần, chính vì thế, đôi khi họ sẽ không thể nhớ hết thuốc nào dùng khi nào, và liệu có tương tác xấu khi dùng nhiều loại thuốc cho các loại bệnh khác nhau hay không.

Đồng thời, bệnh nhân có nhiều mối quan tâm cũng như nhiều chế độ ăn uống sinh hoạt thì cũng khó mà tuân theo nghiêm ngặt khuyến nghị của bác sĩ, dẫn đến rủi ro lớn về các tương tác thuốc - thực phẩm bất lợi mà họ gặp phải.
           """)
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
        Một [bài nghiên cứu](https://jppres.com/jppres/outpatient-prescription-drug-interactions-in-vietnam/) của một nhóm sinh viên Cần Thơ tại một khoa ngoại trú của một trung tâm y tế liên kết với thành phố từ tháng 1 đến tháng 6 năm 2023 chỉ ra rằng có đến **36,7%** đơn thuốc tức là hơn 1/3 đơn được kê chứa **tương tác thuốc bất lợi**.
        """)
    
st.markdown("""
### 🎯 Mục tiêu
**SmartRx** sẽ hỗ trợ lưu lại hết lịch sử dùng thuốc và bệnh tương ứng cho bệnh nhân, từ đó sẽ phát hiện ra các tương tác thuốc bất lợi. Ứng dụng này đồng thời sẽ tổng hợp và đưa ra những gợi ý cho cả bệnh nhân, nhắc nhở bệnh nhân uống thuốc. Kiểm tra tương tác thuốc - thuốc hay thuốc - thực phẩm một cách đơn giản, tiện lợi.

SmartRx đóng vai trò như một sổ khám bệnh điện tử cho bệnh nhân và mọi đối tượng - thao tác dễ dàng, hiệu quả. 
""")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
