from typing import Tuple, Dict
import streamlit as st
import requests
import tomllib 

with open(".streamlit/api_config.toml", "rb") as f:
    config = tomllib.load(f)

api_key = config["google"]["google_gemini_api"]
json_drug_list_path = "smartrx_web/json/drug_list_info.json"
json_drug_check_path = "smartrx_web/json/drug_check_info.json"
json_food_check_path = "smartrx_web/json/food_check_info.json"

url_api = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
def get_drug_info(drug_name: str) -> Tuple[str, Dict]:
    headers = {
        "Content-Type": "application/json",
    }
    
    prompt = (
        f"Vui lòng cung cấp thông tin sơ lược ngắn gọn nhất có thể cho thuốc có tên '{drug_name}'(Nếu không tìm được thuốc tên như vậy thì tự động thay bằng tên thuốc gần giống nhất). "
        "Trả về 3 đoạn văn, mỗi đoạn đánh số như sau, không yêu cầu tiêu đề:"
        "1. (Chỉ có tên thuốc)"
        "2. (Mục đích sử dụng)"
        "3. (Hướng dẫn sử dụng)"
        "Vui lòng đảm bảo các thông tin rõ ràng và chính xác. Chỉ chả về chính xác 3 đoạn văn, không trả về thêm bất cứ text ngoại lệ nào khác"
    )
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url_api, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            candidates = res_json["candidates"]
            if candidates:
                candidate = candidates[0]
                parts_list = candidate["content"]["parts"]
                if not parts_list:
                    st.error("Phần 'parts' không có dữ liệu.")
                    return {}
                full_text = parts_list[0]["text"].strip()
                
                parts = full_text.split("2.")
                drug_name_text = parts[0].strip()[2:]
                
                parts = parts[1].split("3.")
                purpose = parts[0].strip()
                general_usage = parts[1].strip() if len(parts) > 1 else ""
                
                result = {
                    "drug_name": drug_name_text,
                    "purpose": purpose,
                    "general_usage": general_usage
                }
                return drug_name_text, result
            else:
                st.error("API không trả về kết quả hợp lệ.")
                return {}
        else:
            st.error(f"Lỗi từ API: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        st.error(f"Exception khi gọi API Gemini: {str(e)}")
        return {}
    
def get_food_info(food_name: str) -> Tuple[str, Dict]:
    headers = {
        "Content-Type": "application/json",
    }
    
    prompt = (
        f"Vui lòng cung cấp thông tin sơ lược ngắn gọn nhất có thể cho loại thực phẩm có tên '{food_name}'(Nếu không tìm được thuốc tên như vậy thì tự động thay bằng tên thực phẩm gần giống nhất). "
        "Trả về 2 đoạn văn, mỗi đoạn đánh số như sau, không yêu cầu tiêu đề:"
        "1. (Chỉ có tên loại thực phẩm)"
        "2. (Các chất có trong loại thực phẩm đó, yêu cầu có đủ tên chất)"
        "Vui lòng đảm bảo các thông tin rõ ràng và chính xác. Chỉ chả về chính xác 2 đoạn văn, không trả về thêm bất cứ text ngoại lệ nào khác"
    )
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url_api, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            candidates = res_json["candidates"]
            if candidates:
                candidate = candidates[0]
                parts_list = candidate["content"]["parts"]
                if not parts_list:
                    st.error("Phần 'parts' không có dữ liệu.")
                    return {}
                full_text = parts_list[0]["text"].strip()
                
                parts = full_text.split("2.")
                drug_name_text = parts[0].strip()[2:]
                
                food_information = parts[1].strip() if len(parts) > 1 else ""
                
                result = {
                    "drug_name": drug_name_text,
                    "food_information": food_information
                }
                return drug_name_text, result
            else:
                st.error("API không trả về kết quả hợp lệ.")
                return {}
        else:
            st.error(f"Lỗi từ API: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        st.error(f"Exception khi gọi API Gemini: {str(e)}")
        return {}