import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = "여주보_예측_1차통합_데이터셋(26).csv"  # 파일명을 확인하세요
df = pd.read_csv(file_path, encoding='utf-8-sig')

# '시간' 열을 datetime 객체로 변환 (시각화 시 x축을 예쁘게 만들기 위함)
if '시간' in df.columns:
    df['시간'] = pd.to_datetime(df['시간'])
    df.set_index('시간', inplace=True)

# 2. 시각화 설정
cols = df.columns
n_cols = len(cols)

# 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 크기 설정 (열이 많으므로 세로로 길게 설정)
fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=(15, 2 * n_cols), sharex=True)
fig.suptitle('여주보 및 인근 관측소 데이터 통합 시계열 차트', fontsize=20, y=1.01)

# 3. 각 열별로 그래프 그리기
for i, col in enumerate(cols):
    axes[i].plot(df.index, df[col], color='steelblue', lw=1)
    axes[i].set_ylabel(col, rotation=0, labelpad=80, fontsize=10, ha='right')
    axes[i].grid(True, alpha=0.3)

    # y축 범위가 너무 작으면 소수점까지 표시
    axes[i].tick_params(axis='y', labelsize=8)

# 4. 레이아웃 조정 및 저장
plt.tight_layout()
output_filename = "data_full_inspection.png"
plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()

print(f"✅ 시각화 완료! '{output_filename}' 파일로 저장되었습니다.")