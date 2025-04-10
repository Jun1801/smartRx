import streamlit as st
import json
import os

from smartrx_web.utils import *

if not os.path.exists(json_drug_list_path):
    with open(json_drug_list_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

drug_name = st.text_input("Nhập tên thuốc:", "VD: Ibuprofen")

if st.button("Thêm vào danh sách"):
    if drug_name:
        name, drug_info = get_drug_info(drug_name)
        drug_name = name.strip()
        if drug_info:
            with open(json_drug_list_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            data[drug_name] = drug_info
            
            with open(json_drug_list_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            st.success(f"Đã thêm {drug_name} vào danh sách.")
        else:
            st.error("Không lấy được dữ liệu từ API.")
    else:
        st.error("Vui lòng nhập lại tên thuốc.")

with open(json_drug_list_path, "r", encoding="utf-8") as f:
    drugs_data = json.load(f)

if drugs_data:
    for drug_name, info in drugs_data.items():
        with st.expander(drug_name):
            st.markdown(f"**Tên thuốc:** {info.get('drug_name', drug_name)}")
            st.markdown(f"**Mục đích sử dụng:** {info.get('purpose', 'Chưa cập nhật')}")
            st.markdown(f"**Hướng dẫn sử dụng:** {info.get('general_usage', 'Chưa cập nhật')}")
        
            if st.button(f"Xoá khỏi danh sách", key=f"delete_{drug_name}"):
                drugs_data.pop(drug_name)

                with open(json_drug_list_path, "w", encoding="utf-8") as f:
                    json.dump(drugs_data, f, ensure_ascii=False, indent=4)

                st.rerun()
else:
    st.info("Chưa có thông tin thuốc nào được thêm vào.")
