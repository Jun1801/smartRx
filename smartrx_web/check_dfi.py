import streamlit as st
import json
import os

from smartrx_web.utils import *

if not os.path.exists(json_food_check_path):
    with open(json_food_check_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

st.markdown("Bạn muốn ăn gì đó nhưng lại lo ngại sẽ ảnh hưởng đến việc uống thuốc? **SmartRx** sẽ giúp bạn kiểm tra tương tác giữa thực phẩm bạn muốn ăn với các loại thuốc bạn đang sử dụng!")
food_name = st.text_input("Nhập tên thực phẩm:", "VD: Cam")

if st.button("Thêm thực phẩm"):
    if food_name:
        name, food_info = get_food_info(food_name)
        food_name = name.strip()
        if food_info:
            with open(json_food_check_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            data[food_name] = food_info
            
            with open(json_food_check_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            st.success(f"Đã thêm {food_name}.")
        else:
            st.error("Không lấy được dữ liệu từ API.")
    else:
        st.error("Vui lòng nhập lại tên thực phẩm.")

with open(json_food_check_path, "r", encoding="utf-8") as f:
    foods_data = json.load(f)

if foods_data:
    for food_name, info in foods_data.items():
        with st.expander(food_name):
            st.markdown(f"**Tên thực phẩm:** {info.get('food_name', food_name)}")
            st.markdown(f"**Thông tin thực phẩm:** {info.get('food_information', 'Chưa cập nhật')}")

            if st.button(f"Xoá", key=f"delete_{food_name}"):
                foods_data.pop(food_name)

                with open(json_food_check_path, "w", encoding="utf-8") as f:
                    json.dump(foods_data, f, ensure_ascii=False, indent=4)

                st.rerun()
else:
    st.info("Chưa có thông tin thực phẩm nào được thêm vào.")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
