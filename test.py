import requests

# 1. 지원님의 키와 비밀을 여기에 넣으세요
API_KEY = "TWe_Rzp1LBYHAKcImO65mgMuxwej8tG3"
API_SECRET = "A47C3Dxu_PXKKTUuXKv0218XKVdG34Tz"

# 2. 테스트용 이미지 URL (무료 이미지 사이트의 얼굴 사진입니다)
image_url = "https://cdn.pixabay.com/photo/2016/11/29/06/08/woman-1867715_1280.jpg"

url = "https://api-us.faceplusplus.com/facepp/v3/detect"

# 3. 요청 데이터 구성 (사진 주소와 피부 분석 항목 추가)
data = {
    "api_key": API_KEY,
    "api_secret": API_SECRET,
    "image_url": image_url,
    "return_attributes": "skinstatus,gender,age"
}

response = requests.post(url, data=data)

# 4. 결과 출력
print(response.json())