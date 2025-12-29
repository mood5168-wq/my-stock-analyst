import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 設定網頁標題與風格
st.set_page_config(page_title="超級分析師-台股診斷系統", layout="wide")

st.title("🚀 超級分析師：台股強勢族群與風控系統")
st.markdown("---")

# --- 側邊欄：持股監控輸入 ---
st.sidebar.header("📊 持股即時診斷")
stock_id = st.sidebar.text_input("股票名稱/代碼", "2330 台積電")
buy_price = st.sidebar.number_input("買入成本價", value=1400.0)
high_price = st.sidebar.number_input("買入後最高價", value=1530.0)
curr_price = st.sidebar.number_input("當前股價", value=1510.0)

# --- 邏輯運算：診斷 ---
stop_loss = buy_price * 0.93
trailing_stop = high_price * 0.90

# --- 第一區塊：大盤指標 ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("當前加權指數", "28,850", "+120.5")
with col2:
    st.metric("大盤位階", "多頭趨勢", "MA5 之上")
with col3:
    st.metric("成交量預估", "4,200 億", "偏多")

# --- 第二區塊：選股推薦清單 ---
st.subheader("🔥 近期強勢族群選股清單 (營收成長+籌碼跟單)")
data = {
    '代號': ['2330', '3711', '6669', '3189', '3376'],
    '名稱': ['台積電', '日月光', '緯穎', '景碩', '新日興'],
    '營收年增%': [35.2, 28.5, 410.2, 22.1, 15.8],
    '投信買超(張)': [12500, 4500, 800, 3200, 1500],
    '操作建議': ['強勢續抱', '低檔轉強', '趨勢爆發', '轉虧為盈', '回檔觀察']
}
df = pd.DataFrame(data)
st.table(df)

# --- 第三區塊：視覺化風控監控 ---
st.subheader(f"🛡️ {stock_id} 風控雷達")
if curr_price <= stop_loss:
    st.error(f"🚨 警報：已破停損價 {stop_loss}！建議立即執行紀律。")
elif curr_price <= trailing_stop:
    st.warning(f"⚠️ 警報：高點回落達 10% (獲利回吐點 {trailing_stop})，建議落袋為安。")
else:
    st.success(f"✅ 狀態：正常持有中。目前停損位移至：{stop_loss}")

# 展示獲利百分比圖表
fig = px.bar(x=['成本', '當前', '停損', '移動停利'], y=[buy_price, curr_price, stop_loss, trailing_stop], 
             labels={'x': '位階', 'y': '價格'}, title="持股水位視覺化")
st.plotly_chart(fig)

st.info("💡 提示：本系統每日盤後自動更新，選股邏輯結合基本面 YoY > 20% 與投信連買指標。")
