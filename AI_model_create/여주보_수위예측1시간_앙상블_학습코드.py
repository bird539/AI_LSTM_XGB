import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# [수정] tensorflow.keras 대신 keras를 직접 사용
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Input # Input 레이어 추가
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
df = pd.read_csv("여주보_최종_정규화_데이터셋.csv", encoding='utf-8-sig')

target_col = '여주보(상류)_수위_수위(m)'

df['target_future'] = df[target_col].shift(-6)

df = df.dropna()

# 데이터 분리
X = df.drop(columns=['시간', 'target_future'])
y = df['target_future']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- [모델 학습: XGBoost] ---
print("🚀 XGBoost 학습 시작...")
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    n_jobs=-1,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_preds = xgb_model.predict(X_test)

# --- [모델 학습: LSTM (Keras 3 최적화)] ---
print("🚀 LSTM 학습 시작...")
X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# [수정] Keras 3 스타일: Input 레이어를 명시적으로 선언하여 UserWarning 방지
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])), # 입력층 분리
    LSTM(64, return_sequences=True), # input_shape 인자 제거
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

# [수정] verbose=1로 설정하여 학습 진행 상황(Epoch)을 확인하는 것을 권장합니다.
lstm_model.fit(
    X_train_lstm, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_test_lstm, y_test),
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

# 앙상블
final_preds = (xgb_preds + lstm_preds) / 2

# --- [성능 평가 및 역정규화] ---
def calculate_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

# 스케일러 로드 시 에러 방지를 위해 가급적 현재 환경에서 저장된 scaler 사용 권장
target_scaler = joblib.load('target_scaler.pkl')
y_test_real = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
xgb_real = target_scaler.inverse_transform(xgb_preds.reshape(-1, 1)).flatten()
lstm_real = target_scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()
ensemble_real = target_scaler.inverse_transform(final_preds.reshape(-1, 1)).flatten()

# --- [모델 저장] ---
joblib.dump(xgb_model, 'yeoju_xgb_model_1h.pkl')
# [중요] Keras 3에서는 .keras 확장자가 가중치와 구조를 모두 포함하는 표준입니다.
lstm_model.save('yeoju_lstm_model_1h.keras')

print("\n✅ 모델 저장 완료 (Keras 3 규격 적용).")

# --- [성능 결과 지표 출력] ---
print("\n" + "="*50)
print("📊 모델 성능 평가 결과 (역정규화 기준)")
print("-"*50)

def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nse = calculate_nse(y_true, y_pred)
    print(f"[{name}]")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE : {mae:.4f}")
    print(f"   - R²  : {r2:.4f}")
    print(f"   - NSE : {nse:.4f}")
    print("-"*30)

print_metrics("XGBoost", y_test_real, xgb_real)
print_metrics("LSTM", y_test_real, lstm_real)
print_metrics("앙상블(XGB+LSTM)", y_test_real, ensemble_real)
print("="*50)

# --- [성능 결과 시각화] ---
plt.figure(figsize=(16, 8))

# 1. 전체 테스트 구간 비교
plt.subplot(2, 1, 1)
plt.plot(y_test_real, label='실측치 (Actual)', color='black', alpha=0.6, linewidth=2)
plt.plot(ensemble_real, label='앙상블 예측 (Ensemble)', color='red', linestyle='--', alpha=0.8)
plt.title('여주보 상류 수위 예측: 전체 테스트 구간 (역정규화)', fontsize=15)
plt.ylabel('수위 (m)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 마지막 300 시점 확대 (세부 패턴 확인용)
plt.subplot(2, 1, 2)
plt.plot(y_test_real[-300:], label='실측치 (Actual)', color='black', alpha=0.7, linewidth=2)
plt.plot(xgb_real[-300:], label='XGBoost 예측', color='blue', linestyle=':', alpha=0.6)
plt.plot(lstm_real[-300:], label='LSTM 예측', color='green', linestyle=':', alpha=0.6)
plt.plot(ensemble_real[-300:], label='앙상블 예측', color='red', linewidth=2)
plt.title('최근 300 시점 상세 비교', fontsize=15)
plt.xlabel('시간 (Time Steps)')
plt.ylabel('수위 (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- [추가: 오차 분포 히스토그램] ---
plt.figure(figsize=(10, 5))
error = ensemble_real - y_test_real
sns.histplot(error, kde=True, color='purple')
plt.title('앙상블 모델 예측 오차(Error) 분포', fontsize=13)
plt.xlabel('오차 (m)')
plt.ylabel('빈도')
plt.axvline(x=0, color='red', linestyle='--')
plt.show()