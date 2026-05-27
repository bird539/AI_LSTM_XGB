import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from collections import deque  # 속도 최적화를 위한 데크

#서버 실행문 - 터미널에서 다음 입력
#uvicorn AI_API_SERVER:app --reload

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_full_path(filename):
    return os.path.join(BASE_DIR, filename)


# --- [1. API 데이터 규격 정의] ---
class WaterLevelData(BaseModel):
    시간: str
    여주보_상류_수위_수위_m: float
    여주보_상류_수위_유량_m3_s: float
    여주보_상류_수위_해발수위_El_m: float
    여주보_하류_수위_수위_m: float
    여주보_하류_수위_유량_m3_s: float
    여주보_하류_수위_해발수위_El_m: float
    여주시_여주대교_강수량_강수량_mm: float
    여주시_여주대교_강수량_누적강수량_mm: float
    여주시_여주대교_수위_수위_m: float
    여주시_여주대교_수위_유량_m3_s: float
    여주시_여주대교_수위_해발수위_El_m: float
    여주시_주암리_강수량_강수량_mm: float
    여주시_주암리_강수량_누적강수량_mm: float
    원주시_문막교_수위_수위_m: float
    원주시_문막교_수위_유량_m3_s: float
    원주시_문막교_수위_해발수위_El_m: float
    충주댐_댐_현재수위_EL_m: float
    충주댐_댐_유입량_m3_s: float
    충주댐_댐_방류량_m3_s: float
    충주시_수안보면사무소_강수량_강수량_mm: float
    충주시_수안보면사무소_강수량_누적강수량_mm: float
    충주조정지댐_댐_현재수위_EL_m: float
    충주조정지댐_댐_유입량_m3_s: float
    충주조정지댐_댐_방류량_m3_s: float


# --- [2. 모델 및 데이터 로드] ---
app = FastAPI(title="Yeoju Water Level Prediction Server")

try:
    scaler = joblib.load(get_full_path('total_scaler.pkl'))
    target_scaler = joblib.load(get_full_path('target_scaler.pkl'))

    # LSTM 모델 로드 (compile=False로 오버헤드 감소)
    #10분 모델
    model_lstm_10m = load_model(get_full_path('yeoju_lstm_model.keras'), compile=False, safe_mode=False)
    model_xgb_10m = joblib.load(get_full_path('yeoju_xgb_model.pkl'))

    # 1시간 모델 (추가)
    model_lstm_1h = load_model(get_full_path('yeoju_lstm_model_1h.keras'), compile=False, safe_mode=False)
    model_xgb_1h = joblib.load(get_full_path('yeoju_xgb_model_1h.pkl'))

    # 3시간 모델 (추가)
    model_lstm_3h = load_model(get_full_path('yeoju_lstm_model_3h.keras'), compile=False, safe_mode=False)
    model_xgb_3h = joblib.load(get_full_path('yeoju_xgb_model_3h.pkl'))

    model_features = list(scaler.feature_names_in_)
    print("🚀 모든 시점(10m, 1h, 3h) AI 모델 탑재 완료!")
except Exception as e:
    print(f"❌ 로드 오류: {e}")


# history_db를 deque로 설정하여 pop(0) 병목 현상 제거 (최대 100개 유지)
def load_recent_history():
    try:
        db_file = get_full_path("여주보_예측_1차통합_데이터셋(20~25).csv")
        df = pd.read_csv(db_file).tail(100)
        return deque(df.to_dict('records'), maxlen=100)
    except:
        return deque(maxlen=100)


history_db = load_recent_history()


# --- [3. 예측 로직 함수] ---
def perform_multi_prediction(current_data_dict):
    global history_db
    try:
        # 1. 키 매핑
        mapped_dict = {
            "시간": current_data_dict['시간'],
            "여주보(상류)_수위_수위(m)": current_data_dict['여주보_상류_수위_수위_m'],
            "여주보(상류)_수위_유량(m³/s)": current_data_dict['여주보_상류_수위_유량_m3_s'],
            "여주보(상류)_수위_해발수위(El.m)": current_data_dict['여주보_상류_수위_해발수위_El_m'],
            "여주보(하류)_수위_수위(m)": current_data_dict['여주보_하류_수위_수위_m'],
            "여주보(하류)_수위_유량(m³/s)": current_data_dict['여주보_하류_수위_유량_m3_s'],
            "여주보(하류)_수위_해발수위(El.m)": current_data_dict['여주보_하류_수위_해발수위_El_m'],
            "여주시(여주대교)_강수량_강수량(mm)": current_data_dict['여주시_여주대교_강수량_강수량_mm'],
            "여주시(여주대교)_강수량_누적강수량(mm)": current_data_dict['여주시_여주대교_강수량_누적강수량_mm'],
            "여주시(여주대교)_수위_수위(m)": current_data_dict['여주시_여주대교_수위_수위_m'],
            "여주시(여주대교)_수위_유량(m³/s)": current_data_dict['여주시_여주대교_수위_유량_m3_s'],
            "여주시(여주대교)_수위_해발수위(El.m)": current_data_dict['여주시_여주대교_수위_해발수위_El_m'],
            "여주시(주암리)_강수량_강수량(mm)": current_data_dict['여주시_주암리_강수량_강수량_mm'],
            "여주시(주암리)_강수량_누적강수량(mm)": current_data_dict['여주시_주암리_강수량_누적강수량_mm'],
            "원주시(문막교)_수위_수위(m)": current_data_dict['원주시_문막교_수위_수위_m'],
            "원주시(문막교)_수위_유량(m³/s)": current_data_dict['원주시_문막교_수위_유량_m3_s'],
            "원주시(문막교)_수위_해발수위(El.m)": current_data_dict['원주시_문막교_수위_해발수위_El_m'],
            "충주댐_댐_현재수위(EL.m)": current_data_dict['충주댐_댐_현재수위_EL_m'],
            "충주댐_댐_유입량(m³/s)": current_data_dict['충주댐_댐_유입량_m3_s'],
            "충주댐_댐_방류량(m³/s)": current_data_dict['충주댐_댐_방류량_m3_s'],
            "충주시(수안보면사무소)_강수량_강수량(mm)": current_data_dict['충주시_수안보면사무소_강수량_강수량_mm'],
            "충주시(수안보면사무소)_강수량_누적강수량(mm)": current_data_dict['충주시_수안보면사무소_강수량_누적강수량_mm'],
            "충주조정지댐_댐_현재수위(EL.m)": current_data_dict['충주조정지댐_댐_현재수위_EL_m'],
            "충주조정지댐_댐_유입량(m³/s)": current_data_dict['충주조정지댐_댐_유입량_m3_s'],
            "충주조정지댐_댐_방류량(m³/s)": current_data_dict['충주조정지댐_댐_방류량_m3_s'],
        }

        # 2. 시차 피처 생성 (더 효율적인 dict 접근)
        target_cols = ['충주조정지댐_댐_방류량(m³/s)', '원주시(문막교)_수위_유량(m³/s)', '여주시(여주대교)_수위_유량(m³/s)',
                       '충주시(수안보면사무소)_강수량_강수량(mm)', '여주보(상류)_수위_수위(m)']
        lags = [6, 18, 36, 72]

        lag_features = {}
        for col in target_cols:
            for lag in lags:
                if len(history_db) >= lag:
                    val = history_db[-lag].get(col, mapped_dict.get(col, 0))
                else:
                    val = mapped_dict.get(col, 0)
                lag_features[f'{col}_lag_{lag}'] = val

        # 데이터 병합 및 스케일링
        full_dict = {**mapped_dict, **lag_features}
        current_row = pd.DataFrame([full_dict])
        features_for_scaler = current_row.reindex(columns=model_features).fillna(0)
        scaled_data = scaler.transform(features_for_scaler).astype('float32')

        # [핵심] 공통 전처리 함수화 (또는 반복 수행)
        def get_pred(m_lstm, m_xgb, input_scaled):
            xgb_feats = m_xgb.get_booster().feature_names
            df_scaled = pd.DataFrame(input_scaled, columns=model_features)[xgb_feats]

            # LSTM 예측
            l_in = df_scaled.values.reshape((1, 1, len(xgb_feats)))
            p_l = m_lstm(l_in, training=False).numpy()[0][0]

            # XGB 예측
            p_x = m_xgb.predict(df_scaled)[0]

            # 앙상블 및 역정규화
            res = target_scaler.inverse_transform([[(p_l + p_x) / 2]])[0][0]
            return float(res)

        # 각 시점별 예측 수행
        try:
            pred_10m = get_pred(model_lstm_10m, model_xgb_10m, scaled_data)
        except Exception as e:
            print(f"10분 모델 예측 실패: {e}")
            pred_1h = None  # 혹은 0.0
        try:
            pred_1h = get_pred(model_lstm_1h, model_xgb_1h, scaled_data)
        except Exception as e:
            print(f"1시간 모델 예측 실패: {e}")
            pred_1h = None  # 혹은 0.0
        try:
            pred_3h = get_pred(model_lstm_3h, model_xgb_3h, scaled_data)
        except Exception as e:
            print(f"3시간 모델 예측 실패: {e}")
            pred_1h = None  # 혹은 0.0

        # 6. 히스토리 업데이트 (deque가 자동으로 오래된 데이터 삭제)
        history_db.append(mapped_dict)

        return {
            "10m": pred_10m,
            "1h": pred_1h,
            "3h": pred_3h
        }

    except Exception as e:
        print(f"‼️ 에러: {str(e)}")
        raise e


# --- [4. API 엔드포인트] ---
@app.post("/predict")
async def predict(data: WaterLevelData):
    try:
        predictions = perform_multi_prediction(data.dict())
        return {
            "status": "success",
            "predictions": predictions  # {10m: x, 1h: y, 3h: z}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))