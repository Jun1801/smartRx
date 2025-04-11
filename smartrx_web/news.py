import requests
import streamlit as st
import tomllib 

st.markdown("Bạn có thể dễ dàng tìm kiếm, tra cứu thông tin liên quan đến các thuốc, thực phẩm mà bạn cần. Chỉ cần gõ từ khoá và để mọi việc của SmartRx lo!")

with open(".streamlit/api_config.toml", "rb") as f:
    config = tomllib.load(f)

api_key = config["google"]["google_search_api"]
cx = config["google"]["cx"]
query = st.text_input("Nhập từ khóa tìm kiếm:", "tương tác thuốc thuốc")

if query:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "sort": "date"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json().get("items", [])
        if results:
            for item in results:
                st.subheader(item.get("title"))
                st.write(item.get("snippet"))
                st.markdown(f"[Xem chi tiết]({item.get('link')})")
                st.markdown("---")
        else:
            st.warning("Không có kết quả nào được tìm thấy.")
    else:
        st.error(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
