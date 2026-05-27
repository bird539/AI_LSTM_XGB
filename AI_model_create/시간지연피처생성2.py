import pandas as pd
import numpy as np

# 1. 데이터 로드
df = pd.read_csv("여주보_예측_시차적용_최종(26).csv", encoding='utf-8-sig') # 혹은 이전 단계 파일

# 2. 숫자형으로 강제 변환 (문자열이 섞여있을 수 있음)
cols_to_fix = [c for c in df.columns if c != '시간']
df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

# 3. [전략] 지점별 시작점이 다른 문제 해결
# 강수량/유량/방류량 계열: 데이터 시작 전의 공백은 '0'으로 채우는 것이 물리적으로 타당합니다.
# (측정 장비가 없었을 때 비가 안 왔거나 방류가 없었다고 가정하는 방식)
fill_zero_cols = [c for c in df.columns if '강수량' in c or '유량' in c or '방류량' in c]
df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

# 수위 계열: 0으로 채우면 수위가 0m가 되어버려 위험합니다.
# 수위는 해당 열의 '첫 번째 유효한 값'으로 앞부분 공백을 채워줍니다 (bfill).
water_level_cols = [c for c in df.columns if '수위' in c]
df[water_level_cols] = df[water_level_cols].fillna(method='bfill')

# 4. 시차(Lag) 때문에 생긴 아주 미세한 맨 앞 공백 최종 정리
df = df.fillna(0)

# 5. [중요] 여주보 수위 데이터가 실제로 존재하는 기간만 슬라이싱
# (아무리 살려도 예측 대상인 여주보 수위 자체가 없으면 학습이 안 됩니다)
target_y = '여주보(상류)_수위_수위(m)'
# 여주보 수위 데이터가 처음으로 나타나는 인덱스 찾기
first_valid_idx = df[target_y].first_valid_index()
last_valid_idx = df[target_y].last_valid_index()

# 해당 구간만 잘라내기
df_final = df.loc[first_valid_idx:last_valid_idx].reset_index(drop=True)

print(f"📊 보정 후 데이터 총 행수: {len(df_final)}행")
df_final.to_csv("여주보_시간지연피처2_데이터셋(26).csv", index=False, encoding='utf-8-sig')
print("✨ AI 모델에 바로 넣을 수 있는 데이터셋이 완성되었습니다!")