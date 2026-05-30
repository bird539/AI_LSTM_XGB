import requests
import json
from datetime import datetime

# ==========================================
# 0. 사용자 설정 정보
# ==========================================
#한강수문통제소 https://www.hrfco.go.kr/main.do
#키 발급 필요 https://www.hrfco.go.kr/web/openapiPage/certifyKey.do

SERVICE_KEY = "KEY"  # 검증된 인증키
AI_SERVER_URL = "http://127.0.0.1:8000/predict"

# 9개 전체 실시간 관측소 코드 체계 동기화
OBS_CODES = {
    "여주보_상류": "1007639",
    "여주보_하류": "1007641",
    "여주대교_수위": "1007635",
    "여주대교_강수": "10074030",
    "주암리_강수": "10074100",
    "문막교": "1006690",
    "수안보_강수": "10044020",
    "충주댐": "1003110",
    "충주조정지댐": "1003611"
}


# ==========================================
# 1. 한강홍수통제소 최신 10분 데이터 호출 함수
# ==========================================
def fetch_hrfco_latest_data(hydro_type, obs_code):
    url = f"https://api.hrfco.go.kr/{SERVICE_KEY}/{hydro_type}/list/10M/{obs_code}.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            if "content" in res_json and isinstance(res_json["content"], list) and len(res_json["content"]) > 0:
                return res_json["content"][0]
    except Exception:
        pass
    return None


# ==========================================
# 2. 데이터 실시간 추출 및 AI 서버 연동
# ==========================================
def main():
    print("🚀 명세서 맞춤형 10분 단위 실시간 전 관측소 API 수집 시작...\n")

    raw_data_map = {
        "여주보_상류": fetch_hrfco_latest_data("waterlevel", OBS_CODES["여주보_상류"]),
        "여주보_하류": fetch_hrfco_latest_data("waterlevel", OBS_CODES["여주보_하류"]),
        "여주대교_수위": fetch_hrfco_latest_data("waterlevel", OBS_CODES["여주대교_수위"]),
        "여주대교_강수": fetch_hrfco_latest_data("rainfall", OBS_CODES["여주대교_강수"]),
        "주암리_강수": fetch_hrfco_latest_data("rainfall", OBS_CODES["주암리_강수"]),
        "문막교": fetch_hrfco_latest_data("waterlevel", OBS_CODES["문막교"]),
        "수안보_강수": fetch_hrfco_latest_data("rainfall", OBS_CODES["수안보_강수"]),
        "충주댐": fetch_hrfco_latest_data("dam", OBS_CODES["충주댐"]),
        "충주조정지댐": fetch_hrfco_latest_data("dam", OBS_CODES["충주조정지댐"])
    }

    match_report = []

    # 💡 AI 서버 422 에러(float_type 요구) 해결을 위해 결측 시 0.0으로 안전 변환하는 헬퍼 함수
    def verify_and_extract(station_key, data_key, field_name):
        data = raw_data_map.get(station_key)
        obs_code = OBS_CODES.get(station_key, "알 수 없음")

        # 1. API 응답 자체가 없는 경우
        if not data:
            match_report.append({"항목": field_name, "상태": "❌ 수신 실패", "원인": f"관측소[{obs_code}] 응답 없음 -> 0.0 보정", "값": 0.0})
            return 0.0

        clean_data = {str(k).lower().strip(): str(v).strip() for k, v in data.items()}
        target_key = str(data_key).lower().strip()

        # 2. 응답 데이터 내에 키가 없는 경우
        if target_key not in clean_data:
            match_report.append({"항목": field_name, "상태": "❌ 키 누락", "원인": f"응답 내 '{data_key}' 없음 -> 0.0 보정", "값": 0.0})
            return 0.0

        val_str = clean_data[target_key]

        # 3. 데이터가 공백 문자이거나 결측 표시(-)인 경우
        if val_str in ["", "-", "none"]:
            match_report.append({"항목": field_name, "상태": "⚠️ 데이터 공백", "원인": "통제소 서버 결측 -> 0.0 보정 (422방지)", "값": 0.0})
            return 0.0

        # 4. 수치 변환 시도
        try:
            val_float = float(val_str)
            match_report.append({"항목": field_name, "상태": "✅ 성공", "원인": "-", "값": val_float})
            return val_float
        except ValueError:
            match_report.append({"항목": field_name, "상태": "❌ 변환 오류", "원인": f"'{val_str}' 수치 변환 실패 -> 0.0 보정", "값": 0.0})
            return 0.0

    # 실시간 관측 시각 매핑 파싱 (yyyy-MM-dd HH:mm 규격)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    if raw_data_map["여주대교_수위"] and "ymdhm" in raw_data_map["여주대교_수위"]:
        t = str(raw_data_map["여주대교_수위"]["ymdhm"]).strip()
        if len(t) >= 12:
            current_time = f"{t[0:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}"

    # 전송용 Payload 조립 (결측 발생 시 null 대신 0.0 안전 유입)
    payload = {
        "시간": current_time,

        "여주보_상류_수위_수위_m": verify_and_extract("여주보_상류", "wl", "여주보 상류 수위"),
        "여주보_상류_수위_유량_m3_s": verify_and_extract("여주보_상류", "fw", "여주보 상류 유량"),
        "여주보_상류_수위_해발수위_El_m": verify_and_extract("여주보_상류", "wl", "여주보 상류 해발수위"),

        "여주보_하류_수위_수위_m": verify_and_extract("여주보_하류", "wl", "여주보 하류 수위"),
        "여주보_하류_수위_유량_m3_s": verify_and_extract("여주보_하류", "fw", "여주보 하류 유량"),
        "여주보_하류_수위_해발수위_El_m": verify_and_extract("여주보_하류", "wl", "여주보 하류 해발수위"),

        "여주시_여주대교_강수량_강수량_mm": verify_and_extract("여주대교_강수", "rf", "여주대교 강수량"),
        "여주시_여주대교_강수량_누적강수량_mm": verify_and_extract("여주대교_강수", "rf", "여주대교 누적강수량"),
        "여주시_여주대교_수위_수위_m": verify_and_extract("여주대교_수위", "wl", "여주대교 수위"),
        "여주시_여주대교_수위_유량_m3_s": verify_and_extract("여주대교_수위", "fw", "여주대교 유량"),
        "여주시_여주대교_수위_해발수위_El_m": verify_and_extract("여주대교_수위", "wl", "여주대교 해발수위"),

        "여주시_주암리_강수량_강수량_mm": verify_and_extract("주암리_강수", "rf", "주암리 강수량"),
        "여주시_주암리_강수량_누적강수량_mm": verify_and_extract("주암리_강수", "rf", "주암리 누적강수량"),

        "원주시_문막교_수위_수위_m": verify_and_extract("문막교", "wl", "문막교 수위"),
        "원주시_문막교_수위_유량_m3_s": verify_and_extract("문막교", "fw", "문막교 유량"),
        "원주시_문막교_수위_해발수위_El_m": verify_and_extract("문막교", "wl", "문막교 해발수위"),

        "충주댐_댐_현재수위_EL_m": verify_and_extract("충주댐", "swl", "충주댐 현재수위"),
        "충주댐_댐_유입량_m3_s": verify_and_extract("충주댐", "inf", "충주댐 유입량"),
        "충주댐_댐_방류량_m3_s": verify_and_extract("충주댐", "tototf", "충주댐 방류량"),

        "충주시_수안보면사무소_강수량_강수량_mm": verify_and_extract("수안보_강수", "rf", "수안보 강수량"),
        "충주시_수안보면사무소_강수량_누적강수량_mm": verify_and_extract("수안보_강수", "rf", "수안보 누적강수량"),

        "충주조정지댐_댐_현재수위_EL_m": verify_and_extract("충주조정지댐", "swl", "충주조정지댐 현재수위"),
        "충주조정지댐_댐_유입량_m3_s": verify_and_extract("충주조정지댐", "inf", "충주조정지댐 유입량"),
        "충주조정지댐_댐_방류량_m3_s": verify_and_extract("충주조정지댐", "tototf", "충주조정지댐 방류량")
    }

    # ==========================================
    # 📌 [현황 리포트 모니터 출력]
    # ==========================================
    print("==========================================================================")
    print(f"📡 [실시간 10M 데이터 매칭 결과] 관측 수집기준 시각: {current_time}")
    print("==========================================================================")
    print(f"{'조사 대상 필드':<22} | {'파싱 상태':<10} | {'출력값':<12} | {'상세 원인'}")
    print("--------------------------------------------------------------------------")

    fail_count = 0
    for report in match_report:
        status = report["상태"]
        if "❌" in status or "⚠️" in status:
            fail_count += 1
        print(f"{report['항목']:<22} | {status:<10} | {str(report['값']):<12} | {report['원인']}")

    print("==========================================================================")
    print(f"🕵️‍♂️ 안전 조치 완료: 결측된 {fail_count}개의 필드를 0.0 수치형 패딩 처리하여 422 에러를 방어했습니다.")
    print("==========================================================================\n")

    print("📊 AI 전송 Payload JSON 구조:")
    print(json.dumps(payload, ensure_ascii=False, indent=4))
    print("-" * 60)

    # ==========================================
    # 3. AI 예측 서버 데이터 전송
    # ==========================================
    print("🤖 AI 모델 예측 서버로 Real-time Payload를 발송합니다...")
    try:
        response = requests.post(AI_SERVER_URL, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ 인프런스 처리 성공!")
            print(f"📝 AI 연산 결과: {result}")
        else:
            print(f"❌ AI 서버 응답 거절 ({response.status_code}):\n{response.text}")
    except Exception as e:
        print(f"⚠️ AI 예측 백엔드 서버 연결 실패: {e}")


if __name__ == "__main__":
    main()