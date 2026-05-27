import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. 데이터 로드 (시차가 적용된 최종 데이터셋)
# 파일명은 사용자님이 저장하신 실제 파일명으로 수정해주세요.
file_path = "여주보_시간지연피처2_데이터셋(26).csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 2. '시간' 컬럼 분리 및 숫자 데이터 추출
# 정규화는 숫자 열에만 적용해야 합니다.
time_col = df['시간']
data_df = df.drop(columns=['시간'])

# 3. 예측 목표(Target) 컬럼 지정
# 나중에 예측값 역산(Inverse Transform)을 위해 별도로 관리합니다.
target_col_name = '여주보(상류)_수위_수위(m)'

# 4. 전체 데이터 정규화 (MinMaxScaler)
# 0과 1 사이로 모든 피처의 범위를 압축합니다.
total_scaler = MinMaxScaler()
scaled_values = total_scaler.fit_transform(data_df)

# 5. 예측 목표(수위) 전용 스케일러 별도 학습
# 나중에 AI가 내놓은 0.82 같은 소수를 2.45m로 바꾸기 위해 필요합니다.
target_scaler = MinMaxScaler()
target_scaler.fit(data_df[[target_col_name]])

# 6. 정규화된 데이터를 다시 데이터프레임 구조로 복구
df_scaled = pd.DataFrame(scaled_values, columns=data_df.columns)
df_scaled.insert(0, '시간', time_col.values)

# 7. 결과 저장
# 1) 정규화가 완료된 CSV 파일
df_scaled.to_csv("여주보_정규화_데이터셋(26).csv", index=False, encoding='utf-8-sig')

# 2) 전체 피처 스케일러 (실시간 데이터 입력용)
joblib.dump(total_scaler, 'total_scaler.pkl')

# 3) 타겟 전용 스케일러 (예측 결과 해석용)
joblib.dump(target_scaler, 'target_scaler.pkl')

print("✅ 정규화 및 스케일러 저장 완료!")
print(f"📊 총 행수: {len(df_scaled)} / 총 컬럼수: {len(df_scaled.columns)}")
print("📂 저장된 파일: 여주보_정규화_데이터셋.csv, total_scaler.pkl, target_scaler.pkl")