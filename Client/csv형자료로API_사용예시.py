import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from tqdm import tqdm

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- [1. 설정] ---
API_URL = "http://127.0.0.1:8000/predict"
TEST_DATA_PATH = "여주보_예측_1차통합_데이터셋(26).csv"
TARGET_COLUMN = "여주보(상류)_수위_수위(m)"
RESULT_BASE_DIR = "./monthly_test_results"  # 결과가 저장될 메인 폴더

if not os.path.exists(RESULT_BASE_DIR):
    os.makedirs(RESULT_BASE_DIR)

def safe_convert_float(val):
    try:
        res = float(val)
        return res if np.isfinite(res) else 0.0
    except:
        return 0.0

def calculate_nse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))


def run_performance_test():
    # 1. 데이터 로드 및 전처리
    try:
        df = pd.read_csv(TEST_DATA_PATH, encoding='utf-8-sig')
        df['시간'] = pd.to_datetime(df['시간'])
        df = df.sort_values('시간').ffill().fillna(0)

        # [최적화] 시차(target) 컬럼을 월별 루프 밖에서 한 번에 만듭니다.
        # 이렇게 하면 Pandas 경고가 원천 차단되고 속도도 더 빠릅니다.
        df['target_10m'] = df[TARGET_COLUMN].shift(-1)
        df['target_1h'] = df[TARGET_COLUMN].shift(-6)
        df['target_3h'] = df[TARGET_COLUMN].shift(-18)
        print(f"✅ 데이터 로드 완료: 총 {len(df)}행")
    except Exception as e:
        print(f"❌ 로드 오류: {e}")
        return

    # 월별로 데이터 그룹화
    df['year_month'] = df['시간'].dt.to_period('M')
    months = df['year_month'].unique()
    session = requests.Session()

    for month in months:
        month_str = str(month)
        monthly_df = df[df['year_month'] == month].copy()
        print(f"\n📅 {month_str} 테스트 시작 ({len(monthly_df)}행)")

        # 3가지 모델별로 예측값을 저장할 리스트 생성
        results = {
            "10m": {"actual": [], "pred": []},
            "1h": {"actual": [], "pred": []},
            "3h": {"actual": [], "pred": []}
        }

        # 월별 루프 (tqdm 진행바 표시)
        for _, row in tqdm(monthly_df.iterrows(), total=len(monthly_df), desc=f"Sending {month_str}"):
            payload = {
                "시간": str(row["시간"]),
                "여주보_상류_수위_수위_m": safe_convert_float(row["여주보(상류)_수위_수위(m)"]),
                "여주보_상류_수위_유량_m3_s": safe_convert_float(row["여주보(상류)_수위_유량(m³/s)"]),
                "여주보_상류_수위_해발수위_El_m": safe_convert_float(row["여주보(상류)_수위_해발수위(El.m)"]),
                "여주보_하류_수위_수위_m": safe_convert_float(row["여주보(하류)_수위_수위(m)"]),
                "여주보_하류_수위_유량_m3_s": safe_convert_float(row["여주보(하류)_수위_유량(m³/s)"]),
                "여주보_하류_수위_해발수위_El_m": safe_convert_float(row["여주보(하류)_수위_해발수위(El.m)"]),
                "여주시_여주대교_강수량_강수량_mm": safe_convert_float(row["여주시(여주대교)_강수량_강수량(mm)"]),
                "여주시_여주대교_강수량_누적강수량_mm": safe_convert_float(row["여주시(여주대교)_강수량_누적강수량(mm)"]),
                "여주시_여주대교_수위_수위_m": safe_convert_float(row["여주시(여주대교)_수위_수위(m)"]),
                "여주시_여주대교_수위_유량_m3_s": safe_convert_float(row["여주시(여주대교)_수위_유량(m³/s)"]),
                "여주시_여주대교_수위_해발수위_El_m": safe_convert_float(row["여주시(여주대교)_수위_해발수위(El.m)"]),
                "여주시_주암리_강수량_강수량_mm": safe_convert_float(row["여주시(주암리)_강수량_강수량(mm)"]),
                "여주시_주암리_강수량_누적강수량_mm": safe_convert_float(row["여주시(주암리)_강수량_누적강수량(mm)"]),
                "원주시_문막교_수위_수위_m": safe_convert_float(row["원주시(문막교)_수위_수위(m)"]),
                "원주시_문막교_수위_유량_m3_s": safe_convert_float(row["원주시(문막교)_수위_유량(m³/s)"]),
                "원주시_문막교_수위_해발수위_El_m": safe_convert_float(row["원주시(문막교)_수위_해발수위(El.m)"]),
                "충주댐_댐_현재수위_EL_m": safe_convert_float(row["충주댐_댐_현재수위(EL.m)"]),
                "충주댐_댐_유입량_m3_s": safe_convert_float(row["충주댐_댐_유입량(m³/s)"]),
                "충주댐_댐_방류량_m3_s": safe_convert_float(row["충주댐_댐_방류량(m³/s)"]),
                "충주시_수안보면사무소_강수량_강수량_mm": safe_convert_float(row["충주시(수안보면사무소)_강수량_강수량(mm)"]),
                "충주시_수안보면사무소_강수량_누적강수량_mm": safe_convert_float(row["충주시(수안보면사무소)_강수량_누적강수량(mm)"]),
                "충주조정지댐_댐_현재수위_EL_m": safe_convert_float(row["충주조정지댐_댐_현재수위(EL.m)"]),
                "충주조정지댐_댐_유입량_m3_s": safe_convert_float(row["충주조정지댐_댐_유입량(m³/s)"]),
                "충주조정지댐_댐_방류량_m3_s": safe_convert_float(row["충주조정지댐_댐_방류량(m³/s)"])
            }

            try:
                # 31만 행이므로 타임아웃은 5초 내외로 짧게 잡고 넘기는게 나음
                resp = session.post(API_URL, json=payload, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()['predictions']

                    # [핵심] 현재 시점의 예측값을 저장하되,
                    # 나중에 비교할 정답(target_xx)이 존재하는 경우에만 리스트에 넣습니다.
                    if pd.notna(row['target_10m']):
                        results["10m"]["pred"].append(data["10m"])
                        results["10m"]["actual"].append(row['target_10m'])

                    if pd.notna(row['target_1h']):
                        results["1h"]["pred"].append(data["1h"])
                        results["1h"]["actual"].append(row['target_1h'])

                    if pd.notna(row['target_3h']):
                        results["3h"]["pred"].append(data["3h"])
                        results["3h"]["actual"].append(row['target_3h'])
            except:
                continue

        # 3. 월별 결과 계산 및 저장
        if len(results["10m"]["pred"]) > 10:
            save_path = os.path.join(RESULT_BASE_DIR, month_str)
            if not os.path.exists(save_path): os.makedirs(save_path)

            plt.figure(figsize=(15, 12))

            report_text = f"📅 Month: {month_str}\n"
            report_text += f"Total Rows: {len(monthly_df)}\n"
            report_text += "=" * 30 + "\n"

            # 모델별 루프를 돌며 그래프와 리포트 작성
            for i, mode in enumerate(["10m", "1h", "3h"], 1):
                y_true = results[mode]["actual"]
                y_pred = results[mode]["pred"]

                r2 = r2_score(y_true, y_pred)
                nse = calculate_nse(y_true, y_pred)  # NSE 추가
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)  # MAE도 함께 보면 좋습니다.

                report_text += f"[{mode} Model]\n"
                report_text += f"NSE : {nse:.4f}\n"
                report_text += f"R2  : {r2:.4f}\n"
                report_text += f"RMSE: {rmse:.4f}\n"
                report_text += f"MAE : {mae:.4f}\n\n"

                # 서브플롯 생성
                plt.subplot(3, 1, i)
                plt.plot(y_true, label='Actual', color='black', alpha=0.4)
                plt.plot(y_pred, label=f'Predicted ({mode})',
                         color='red' if mode == "10m" else ('orange' if mode == "1h" else 'blue'),
                         linestyle='--')
                plt.title(f'{mode} 예측 성능 ({month_str}) - NSE: {nse:.4f}')
                plt.legend()
                plt.grid(True, alpha=0.2)

            # 통합 그래프 저장
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"combined_chart_{month_str}.png"))
            plt.close()

            # 리포트 파일 저장
            with open(os.path.join(save_path, f"report_{month_str}.txt"), "w", encoding='utf-8') as f:
                f.write(report_text)

            print(f"📊 {month_str} 테스트 완료 (10m/1h/3h 통합 리포트 저장)")
        else:
            print(f"⚠️ {month_str} 데이터 부족으로 스킵")

    print(f"\n🏁 모든 월별 테스트 완료! 결과 폴더: {RESULT_BASE_DIR}")


if __name__ == "__main__":
    run_performance_test()