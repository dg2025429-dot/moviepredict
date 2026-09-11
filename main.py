import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

LOW_PRED_THRESHOLD = 1000  # 이보다 작게 예측되면 그래프 바닥에 붙여서 표시
FLOOR_VALUE = 100  # 로그축 바닥에 붙일 때 사용할 값 (0/음수는 로그축에 표시 불가)

FEATURE_OPTIONS = {
    "first_scrn": "첫 관측일 스크린수 (first_scrn)",
    "first_show": "첫 관측일 상영횟수 (first_show)",
    "peak": "성수기 개봉 여부 (peak, 12·1·7·8월=1)",
    "first_week_audi": "첫 주 관객 수 (first_week_audi)",
    "days_in_top10": "10위권 유지 일수 (days_in_top10)",
}

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 박스오피스 데이터를 이용해 영화의 총 관객 수를 예측하는 다중 회귀 모델입니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_raw_text(url: str) -> str:
    resp = requests.get(url)
    resp.encoding = "utf-8"
    return resp.text


@st.cache_data(show_spinner=False)
def parse_csv(raw_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(raw_text))


daily_raw = load_raw_text(DAILY_URL)
movies_raw = load_raw_text(MOVIES_URL)

daily = parse_csv(daily_raw)
movies = parse_csv(movies_raw)

# ---------------------------------------------------------------
# 기간 계산 (일별 박스오피스 표 기준)
# ---------------------------------------------------------------
daily["_날짜_dt"] = pd.to_datetime(daily["날짜"], format="%Y%m%d")
start_date = daily["_날짜_dt"].min().strftime("%Y-%m-%d")
end_date = daily["_날짜_dt"].max().strftime("%Y-%m-%d")

st.subheader("📅 데이터 기준 기간")
st.write(f"일별 박스오피스 데이터 기준 **{start_date} ~ {end_date}**")

# ---------------------------------------------------------------
# 영화별 표의 맨 위 열(헤더) 줄을 그대로 표시
# ---------------------------------------------------------------
st.subheader("📋 영화별 표 헤더 (원본 그대로)")
movies_header_line = movies_raw.lstrip("\ufeff").split("\n")[0].strip()
st.code(movies_header_line, language=None)

# ---------------------------------------------------------------
# 사이드바: 사용할 변수 선택
# ---------------------------------------------------------------
st.sidebar.header("⚙️ 모델에 사용할 변수 선택")
selected_features = []
for col, label in FEATURE_OPTIONS.items():
    if st.sidebar.checkbox(label, value=True, key=f"chk_{col}"):
        selected_features.append(col)

if not selected_features:
    st.warning("최소 한 개 이상의 변수를 사이드바에서 선택해 주세요.")
    st.stop()

# ---------------------------------------------------------------
# 영화코드 순 정렬 후, 열 편마다 앞의 세 편을 테스트용으로 분리
# ---------------------------------------------------------------
movies_sorted = movies.sort_values("movieCd").reset_index(drop=True)
group_pos = movies_sorted.index % 10  # 그룹(10편) 내 위치
is_test = group_pos < 3

target_col = "total_audi"
needed_cols = selected_features + [target_col]

work = movies_sorted.copy()
work["_is_test"] = is_test
work = work.dropna(subset=needed_cols)

train_df = work[~work["_is_test"]]
test_df = work[work["_is_test"]]

n_train = len(train_df)
n_test = len(test_df)

st.subheader("🔢 학습 / 평가 규모")
c1, c2, c3 = st.columns(3)
c1.metric("학습에 사용한 영화 편수", f"{n_train:,}편")
c2.metric("평가에 사용한 영화 편수", f"{n_test:,}편")
c3.metric("기준 기간", f"{start_date} ~ {end_date}")

# ---------------------------------------------------------------
# 다중 회귀 모델 학습 및 평가
# ---------------------------------------------------------------
X_train = train_df[selected_features].values
y_train = train_df[target_col].values
X_test = test_df[selected_features].values
y_test = test_df[target_col].values

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📈 모델 평가 결과 (테스트용 영화 기준)")
m1, m2 = st.columns(2)
m1.metric("R² 점수", f"{r2:.3f}")
m2.metric("평균 절대 오차 (MAE)", f"{mae:,.0f}명")

with st.expander("사용한 회귀식 계수 보기"):
    coef_df = pd.DataFrame(
        {"변수": selected_features, "계수": model.coef_}
    )
    st.dataframe(coef_df, use_container_width=True)
    st.write(f"절편(intercept): {model.intercept_:,.2f}")

# ---------------------------------------------------------------
# 예측이 1,000명보다 작은 영화 처리
# ---------------------------------------------------------------
low_pred_mask = y_pred < LOW_PRED_THRESHOLD
n_low_pred = int(low_pred_mask.sum())

plot_pred = y_pred.copy()
plot_pred[plot_pred <= 0] = FLOOR_VALUE
plot_pred[low_pred_mask] = FLOOR_VALUE  # 바닥에 붙여서 표시

plot_actual = y_test.copy().astype(float)
plot_actual[plot_actual <= 0] = FLOOR_VALUE

st.subheader("🎯 실제 관객 수 vs 예측 관객 수 (테스트 영화)")
st.write(
    f"예측된 총 관객 수가 {LOW_PRED_THRESHOLD:,}명보다 작은 영화는 "
    f"**{n_low_pred}편**이며, 그래프 바닥에 붙여 표시했습니다."
)

axis_min = min(plot_actual.min(), plot_pred.min()) * 0.8
axis_max = max(plot_actual.max(), plot_pred.max()) * 1.2

fig = go.Figure()

normal_mask = ~low_pred_mask
fig.add_trace(
    go.Scatter(
        x=plot_actual[normal_mask],
        y=plot_pred[normal_mask],
        mode="markers",
        name="예측값 (1,000명 이상)",
        marker=dict(size=9, color="#1f77b4", opacity=0.75),
        text=test_df.loc[normal_mask, "movieNm"] if normal_mask.any() else None,
        hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>",
    )
)

if low_pred_mask.any():
    fig.add_trace(
        go.Scatter(
            x=plot_actual[low_pred_mask],
            y=plot_pred[low_pred_mask],
            mode="markers",
            name=f"예측 <1,000명 (바닥 표시, {n_low_pred}편)",
            marker=dict(size=9, color="#d62728", symbol="triangle-down", opacity=0.85),
            text=test_df.loc[low_pred_mask, "movieNm"],
            hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>실제 예측값: 1,000명 미만<extra></extra>",
        )
    )

fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="실제 = 예측 (대각선)",
        line=dict(color="gray", dash="dash"),
    )
)

fig.update_xaxes(type="log", title="실제 총 관객 수 (로그)", range=[np.log10(axis_min), np.log10(axis_max)])
fig.update_yaxes(type="log", title="예측 총 관객 수 (로그)", range=[np.log10(axis_min), np.log10(axis_max)])
fig.update_layout(
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("테스트 영화 상세 결과 보기"):
    detail_df = test_df[["movieCd", "movieNm"]].copy()
    detail_df["실제 총 관객 수"] = y_test
    detail_df["예측 총 관객 수"] = y_pred.round(0)
    detail_df["오차(실제-예측)"] = (y_test - y_pred).round(0)
    st.dataframe(detail_df.reset_index(drop=True), use_container_width=True)
