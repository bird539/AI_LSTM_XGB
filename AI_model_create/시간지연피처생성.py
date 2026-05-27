import pandas as pd
import numpy as np

# 1. 데이터 로드
df = pd.read_csv("여주보_예측_1차통합_데이터셋(26).csv", encoding='utf-8-sig')

# [체크] 만약 첫 번째 행이 데이터가 아니라 컬럼명과 똑같은 문자열이라면 제거
# (데이터 합치기 과정에서 중복 생성되었을 경우를 대비)
target_y = '여주보(상류)_수위_수위(m)'
df = df[df[target_y] != target_y]

# 2. 숫자 변환 (중요: 문자열 "-"나 "열이름" 등을 모두 결측치로 처리)
cols_to_fix = [c for c in df.columns if c != '시간']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. 시차(Lag) 생성
target_cols = [
    '충주조정지댐_댐_방류량(m³/s)',
    '원주시(문막교)_수위_유량(m³/s)',
    '여주시(여주대교)_수위_유량(m³/s)',
    '충주시(수안보면사무소)_강수량_강수량(mm)',
    '여주보(상류)_수위_수위(m)'
]
lags = [6, 18, 36, 72]

for col in target_cols:
    if col in df.columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# 4. 결측치 처리 (0행 방지 전략)
# 강수량 계열은 0으로
rf_cols = [c for c in df.columns if '강수량' in c]
df[rf_cols] = df[rf_cols].fillna(0)

# 수위/댐 계열은 보간법으로 (너무 큰 공백은 채우지 않음)
df[cols_to_fix] = df[cols_to_fix].interpolate(method='linear', limit_direction='both')

# 5. [핵심 수정] dropna()를 쓰지 말고 '여주보 수위'가 있는 행만 필터링
# 특정 지점(예: 수안보) 데이터가 통째로 없더라도 전체가 삭제되지 않게 함
df_final = df.dropna(subset=[target_y])

print(f"📊 최종 데이터 보존 성공: {len(df_final)}행")

if len(df_final) > 0:
    df_final.to_csv("여주보_예측_시차적용_최종(26).csv", index=False, encoding='utf-8-sig')
    print("✨ '여주보_예측_시차적용_최종(26).csv'로 저장되었습니다.")
else:
    print("❌ 여전히 0행입니다. 데이터에 심각한 결함이 있습니다.")
    # 원인 파악을 위한 출력
    print(df[target_y].describe())