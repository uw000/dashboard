import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

# 타이틀 설정
st.title("해외 주식 데이터 비교 대시보드")

# 사용 가능한 섹터와 종목 리스트 예시
sectors = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH"],
    "Finance": ["JPM", "BAC", "WFC", "C", "GS"],
    "Energy": ["XOM", "CVX", "COP", "PSX", "VLO"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"]
}

# 섹터와 종목 선택 UI
selected_sector = st.selectbox("섹터를 선택하세요", list(sectors.keys()))
selected_stocks = st.multiselect("비교할 종목을 선택하세요", sectors[selected_sector])

# 기간 설정 (지난 1년 데이터)
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# 데이터 수집 함수
def fetch_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        stock_df = yf.download(ticker, start=start_date, end=end_date)
        stock_df["Ticker"] = ticker
        stock_data[ticker] = stock_df
    return stock_data

# 데이터 가져오기
if selected_stocks:
    stock_data = fetch_stock_data(selected_stocks)

    # 주가 데이터 표시
    st.subheader("선택한 종목의 주가 데이터")
    for ticker, data in stock_data.items():
        st.line_chart(data["Close"], width=0, height=300, use_container_width=True)

    # 재무 지표 비교 테이블 생성
    financial_data = {
        "Ticker": [],
        "Average PER": [],
        "Average PBR": [],
        "EV/EBITDA": [],
        "CAGR": []
    }

    for ticker, data in stock_data.items():
        # PER, PBR 등 샘플 데이터 생성 (실제 데이터 API 사용 필요)
        financial_data["Ticker"].append(ticker)
        financial_data["Average PER"].append(np.random.uniform(10, 30))
        financial_data["Average PBR"].append(np.random.uniform(1, 5))
        financial_data["EV/EBITDA"].append(np.random.uniform(8, 20))
        
        # CAGR 계산
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        years = len(data) / 252
        cagr = (end_price / start_price) ** (1 / years) - 1
        financial_data["CAGR"].append(cagr * 100)

    # 재무 지표 데이터프레임
    financial_df = pd.DataFrame(financial_data)
    st.subheader("재무 지표 비교")
    st.dataframe(financial_df)

    # 6개월 후 주가 예측 (간단한 선형 회귀 모델 사용)
    st.subheader("6개월 후 주가 예측")
    prediction_data = []

    for ticker, data in stock_data.items():
        data.reset_index(inplace=True)
        data["Days"] = (data["Date"] - data["Date"].min()).dt.days
        X = data[["Days"]]
        y = data["Close"]

        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(X, y)

        # 6개월 후 예측
        future_days = 252 // 2  # 약 6개월
        future_price = model.predict([[X["Days"].max() + future_days]])[0]
        prediction_data.append({"Ticker": ticker, "6개월 후 예측 주가": future_price})

    # 예측 결과 데이터프레임 표시
    prediction_df = pd.DataFrame(prediction_data)
    st.dataframe(prediction_df)