import requests
import json
from datetime import datetime

# ==========================================
# 0. 사용자 설정 정보
# ==========================================
SERVICE_KEY = "YOUR_REAL_API_KEY"  # 본인의 API 키 입력
AI_SERVER_URL = "http://127.0.0.1:8000/predict"

OBS_CODES = {
    "여주보_상류": "1007639",
    "여주보_하류": "1007641",
    "여주대교_수위": "1007635",
    "여주대교_강수": "10074030",
    "주암리_강수": "10074100",
    "문막교": "1006690"
}


# ==========================================
# 1. 한강홍수통제소 API 호출 함수
# ==========================================
def fetch_hrfco_data(hydro_type, obs_code):
    url = f"https://api.hrfco.go.kr/{SERVICE_KEY}/{hydro_type}/list/1H/{obs_code}.json"
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
# 2. 데이터 수집 및 상태 체크 매칭
# ==========================================
def main():
    print("🚀 한강홍수통제소 실시간 데이터 수집 및 매칭 검증 시작...\n")

    # 데이터 원본 수집
    raw_data_map = {
        "여주보_상류": fetch_hrfco_data("waterlevel", OBS_CODES["여주보_상류"]),
        "여주보_하류": fetch_hrfco_data("waterlevel", OBS_CODES["여주보_하류"]),
        "여주대교_수위": fetch_hrfco_data("waterlevel", OBS_CODES["여주대교_수위"]),
        "여주대교_강수": fetch_hrfco_data("rainfall", OBS_CODES["여주대교_강수"]),
        "주암리_강수": fetch_hrfco_data("rainfall", OBS_CODES["주암리_강수"]),
        "문막교": fetch_hrfco_data("waterlevel", OBS_CODES["문막교"])
    }

    match_report = []  # 매칭 결과를 저장할 리스트

    # 데이터 매칭 및 검증용 헬퍼 함수
    def verify_and_extract(station_key, data_key, field_name):
        data = raw_data_map.get(station_key)
        obs_code = OBS_CODES.get(station_key, "알 수 없음")

        if not data:
            match_report.append({"항목": field_name, "상태": "❌ 수신 실패", "원인": f"관측소[{obs_code}] 응답 없음", "값": 0.0})
            return 0.0

        if data_key not in data or data[data_key] is None:
            match_report.append({"항목": field_name, "상태": "❌ 키 누락", "원인": f"응답 내 '{data_key}' 키 없음", "값": 0.0})
            return 0.0

        val_str = str(data[data_key]).strip()
        if val_str == "" or val_str == "-":
            match_report.append({"항목": field_name, "상태": "⚠️ 데이터 공백", "원인": "서버가 빈 값을 보냄 (공백 처리)", "값": 0.0})
            return 0.0

        try:
            val_float = float(val_str)
            match_report.append({"항목": field_name, "상태": "✅ 성공", "원인": "-", "값": val_float})
            return val_float
        except ValueError:
            match_report.append({"항목": field_name, "상태": "❌ 변환 오류", "원인": f"'{val_str}' 숫자로 변환 불가", "값": 0.0})
            return 0.0

    # 1시간 단위 최신 시간 추출
    current_time = datetime.now().strftime("%Y-%m-%d %H:00")
    if raw_data_map["여주대교_수위"] and "ymdhm" in raw_data_map["여주대교_수위"]:
        t = raw_data_map["여주대교_수위"]["ymdhm"]
        if len(t) >= 10:
            current_time = f"{t[0:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:00"

    # payload 조립과 동시에 각 항목 검증 진행
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

        # 외부 수집 데이터 항목 (임시 유지)
        "충주댐_댐_현재수위_EL_m": 135.5,
        "충주댐_댐_유입량_m3_s": 120.0,
        "충주댐_댐_방류량_m3_s": 150.0,
        "충주시_수안보면사무소_강수량_강수량_mm": 0.0,
        "충주시_수안보면사무소_강수량_누적강수량_mm": 5.0,
        "충주조정지댐_댐_현재수위_EL_m": 65.2,
        "충주조정지댐_댐_유입량_m3_s": 150.0,
        "충주조정지댐_댐_방류량_m3_s": 155.0
    }

    # ==========================================
    # 📌 [결과 출력 리포트] 현황판 생성
    # ==========================================
    print("==========================================================================")
    print(f"📡 [매칭 현황 검증 리포트] 기준시간: {current_time}")
    print("==========================================================================")
    print(f"{'조사 항목':<22} | {'상태':<10} | {'최종 파싱 값':<12} | {'상세 원인'}")
    print("--------------------------------------------------------------------------")

    fail_count = 0
    for report in match_report:
        status = report["상태"]
        if "❌" in status or "⚠️" in status:
            fail_count += 1
        print(f"{report['항목']:<22} | {status:<10} | {report['값']:<12} | {report['원인']}")

    print("==========================================================================")
    if fail_count == 0:
        print("🎉 모든 실시간 데이터가 정상적으로 매칭되었습니다.")
    else:
        print(f"🕵️‍♂️ 총 {fail_count}개의 항목에 이슈(데이터 공백 또는 오류)가 발견되었습니다. 위 내역을 확인하세요.")
    print("==========================================================================\n")

    # ==========================================
    # 3. AI 예측 서버로 데이터 전송
    # ==========================================
    print("🤖 AI 예측 서버로 전송을 시도합니다...")
    try:
        response = requests.post(AI_SERVER_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ 예측 성공! 서버 응답: {response.json()}")
        else:
            print(f"❌ AI 서버 에러: {response.status_code}\n{response.text}")
    except Exception as e:
        print(f"⚠️ AI 서버 연결 실패: {e}")


if __name__ == "__main__":
    main()