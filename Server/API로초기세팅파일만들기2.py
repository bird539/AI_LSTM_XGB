import requests
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 0. 사용자 설정 정보 (인증키 및 관측소)
# ==========================================
SERVICE_KEY = "KEY"
OUTPUT_FILENAME = "여주보_예측_필요_최근100행_데이터셋.csv"

OBS_CODES = {
    "여주보_상류": "1007639",
    "여주보_하류": "1007641",
    "여주대교_수위": "1007635",
    "문막교_수위": "1006690",
    "여주대교_강수": "10074030",
    "주암리_강수": "10074100",
    "수안보_강수": "10044020",
    "충주댐": "1003110",
    "충주조정지댐": "1003611"
}


def fetch_hrfco_data(hydro_type, obs_code, start_dt, end_dt):
    # 명세서 가이드 제공 규격 예시와 100% 동기화된 URL 구조 (.json 확장자 요청)
    url = f"https://api.hrfco.go.kr/{SERVICE_KEY}/{hydro_type}/list/10M/{obs_code}/{start_dt}/{end_dt}.json"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if "content" in res_json and isinstance(res_json["content"], list):
                return res_json["content"]
    except:
        pass
    return []


def main():
    # 💡 10분 단위 동기화 마스터 타임라인 생성
    now = datetime.now()

    # 현재 분을 10분 단위 정각으로 버림 처리 (예: 13:44 -> 13:40)
    round_minute = (now.minute // 10) * 10
    base_end_time = now.replace(minute=round_minute, second=0, microsecond=0)
    base_start_time = base_end_time - timedelta(days=3)

    # API 요청용 ymdhm 문자열 포맷 생성
    sdt = base_start_time.strftime("%Y%m%d%H%M")
    edt = base_end_time.strftime("%Y%m%d%H%M")

    print("============================================================")
    print(f"📅 데이터 요청 시작 시간: {sdt}")
    print(f"📅 데이터 요청 종료 시간: {edt}")
    print("============================================================")
    print("🚀 명세서 맞춤 구조 및 10분 정각 데이터 수집 시작...\n" + "-" * 60)

    # 10분 단위 타임라인 테이블 선언
    time_range = pd.date_range(start=base_start_time, end=base_end_time, freq='10min')
    final_df = pd.DataFrame({"ymdhm": time_range.strftime("%Y%m%d%H%M")})

    # 관측소별 규격 매핑 관계
    station_configs = [
        {"name": "여주보_상류", "type": "waterlevel", "mappings": {"wl": "여주보(상류)_수위_수위(m)", "fw": "여주보(상류)_수위_유량(m³/s)"}},
        {"name": "여주보_하류", "type": "waterlevel", "mappings": {"wl": "여주보(하류)_수위_수위(m)", "fw": "여주보(하류)_수위_유량(m³/s)"}},
        {"name": "여주대교_수위", "type": "waterlevel",
         "mappings": {"wl": "여주시(여주대교)_수위_수위(m)", "fw": "여주시(여주대교)_수위_유량(m³/s)"}},
        {"name": "문막교_수위", "type": "waterlevel", "mappings": {"wl": "원주시(문막교)_수위_수위(m)", "fw": "원주시(문막교)_수위_유량(m³/s)"}},
        {"name": "여주대교_강수", "type": "rainfall", "mappings": {"rf": "여주시(여주대교)_강수량_강수량(mm)"}},
        {"name": "주암리_강수", "type": "rainfall", "mappings": {"rf": "여주시(주암리)_강수량_강수량(mm)"}},
        {"name": "수안보_강수", "type": "rainfall", "mappings": {"rf": "충주시(수안보면사무소)_강수량_강수량(mm)"}},
        {"name": "충주댐", "type": "dam",
         "mappings": {"swl": "충주댐_댐_현재수위(EL.m)", "inf": "충주댐_댐_유입량(m³/s)", "tototf": "충주댐_댐_방류량(m³/s)"}},
        {"name": "충주조정지댐", "type": "dam",
         "mappings": {"swl": "충주조정지댐_댐_현재수위(EL.m)", "inf": "충주조정지댐_댐_유입량(m³/s)", "tototf": "충주조정지댐_댐_방류량(m³/s)"}}
    ]

    for cfg in station_configs:
        raw_list = fetch_hrfco_data(cfg["type"], OBS_CODES[cfg["name"]], sdt, edt)

        if raw_list:
            df = pd.DataFrame(raw_list)
            df.columns = [str(c).lower().strip() for c in df.columns]

            # 공백 정제 처리 후 수치 변환
            for src_col in cfg["mappings"].keys():
                if src_col in df.columns:
                    df[src_col] = df[src_col].astype(str).str.strip()
                    df[src_col] = pd.to_numeric(df[src_col], errors='coerce')

            select_cols = ['ymdhm'] + list(cfg["mappings"].keys())
            df_subset = df[select_cols].copy()
            df_subset = df_subset.rename(columns=cfg["mappings"])

            final_df = pd.merge(final_df, df_subset, on='ymdhm', how='left')
            print(f"  - {cfg['name']}: ✅ 실시간 데이터 병합 완료")
        else:
            for target_col in cfg["mappings"].values():
                final_df[target_col] = None
            print(f"  - {cfg['name']}: ⚠️ 수신된 Raw 데이터가 없습니다.")

        # 보조/누적 필드 복사 매핑
        if cfg["name"] == "여주보_상류":
            final_df['여주보(상류)_수위_해발수위(El.m)'] = final_df['여주보(상류)_수위_수위(m)']
        elif cfg["name"] == "여주보_하류":
            final_df['여주보(하류)_수위_해발수위(El.m)'] = final_df['여주보(하류)_수위_수위(m)']
        elif cfg["name"] == "여주대교_수위":
            final_df['여주시(여주대교)_수위_해발수위(El.m)'] = final_df['여주시(여주대교)_수위_수위(m)']
        elif cfg["name"] == "문막교_수위":
            final_df['원주시(문막교)_수위_해발수위(El.m)'] = final_df['원주시(문막교)_수위_수위(m)']
        elif cfg["name"] == "여주대교_강수":
            final_df['여주시(여주대교)_강수량_누적강수량(mm)'] = final_df['여주시(여주대교)_강수량_강수량(mm)']
        elif cfg["name"] == "주암리_강수":
            final_df['여주시(주암리)_강수량_누적강수량(mm)'] = final_df['여주시(주암리)_강수량_강수량(mm)']
        elif cfg["name"] == "수안보_강수":
            final_df['충주시(수안보면사무소)_강수량_누적강수량(mm)'] = final_df['충주시(수안보면사무소)_강수량_강수량(mm)']

    print("-" * 60)

    # 정렬 (기본값 강제 대체 코드 없음 -> 누락 구간은 공백/NaN 보존)
    final_df = final_df.sort_values(by='ymdhm', ascending=True)

    # 타임스탬프 가독성 정제
    final_df['ymdhm'] = pd.to_datetime(final_df['ymdhm'], format='%Y%m%d%H%M')
    final_df['시간'] = final_df['ymdhm'].dt.strftime('%Y-%m-%d %H:%M')

    all_columns = [
        "시간",
        "여주보(상류)_수위_수위(m)", "여주보(상류)_수위_유량(m³/s)", "여주보(상류)_수위_해발수위(El.m)",
        "여주보(하류)_수위_수위(m)", "여주보(하류)_수위_유량(m³/s)", "여주보(하류)_수위_해발수위(El.m)",
        "여주시(여주대교)_강수량_강수량(mm)", "여주시(여주대교)_강수량_누적강수량(mm)", "여주시(여주대교)_수위_수위(m)",
        "여주시(여주대교)_수위_유량(m³/s)", "여주시(여주대교)_수위_해발수위(El.m)", "여주시(주암리)_강수량_강수량(mm)",
        "여주시(주암리)_강수량_누적강수량(mm)", "원주시(문막교)_수위_수위(m)", "원주시(문막교)_수위_유량(m³/s)",
        "원주시(문막교)_수위_해발수위(El.m)",
        "충주댐_댐_현재수위(EL.m)", "충주댐_댐_유입량(m³/s)", "충주댐_댐_방류량(m³/s)",
        "충주시(수안보면사무소)_강수량_강수량(mm)", "충주시(수안보면사무소)_강수량_누적강수량(mm)",
        "충주조정지댐_댐_현재수위(EL.m)", "충주조정지댐_댐_유입량(m³/s)", "충주조정지댐_댐_방류량(m³/s)"
    ]

    final_df = final_df[all_columns]

    # 최근 100행 추출 및 내보내기
    final_df = final_df.tail(100)

    final_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
    print(f"🎉 명세서 기반 필터링 및 수집 완료: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()