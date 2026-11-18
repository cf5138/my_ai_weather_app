import streamlit as st
import sqlite3
import pandas as pd
import torch
import torch.nn as nn

# 1. 设置网页基础信息
st.set_page_config(page_title="我的AI天气预测系统", page_icon="🌤️")
st.title("🌤️ 株洲天气数据看板与 AI 预测")
st.write("这是一个由 爬虫 + 数据库 + PyTorch + Web 组成的全栈项目！")

try:
    # 2. 从数据库读数据
    conn = sqlite3.connect('weather.db')
    df = pd.read_sql_query("SELECT * FROM daily_weather", conn)
    conn.close()

    # 3. 展示图表和表格
    st.subheader("近期温度趋势")
    # 将日期转换为 datetime 格式，并设置为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 用 Streamlit 自带的折线图展示
    st.line_chart(df)
    
    st.subheader("详细数据")
    st.dataframe(df)

    # 4. AI 预测部分
    temps = df['max_temp'].tolist()
    if len(temps) > 1:
        # 把数据转成张量
        X = torch.tensor(temps[:-1], dtype=torch.float32).view(-1, 1)
        Y = torch.tensor(temps[1:], dtype=torch.float32).view(-1, 1)
        
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 在网页里跑训练，为了速度，直接降到 20 次即可
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
        
        today_temp = torch.tensor([[temps[-1]]], dtype=torch.float32)
        predicted = model(today_temp).item()
        
        st.subheader("🤖 AI 预测结果")
        st.info(f"今天最高温：{temps[-1]}°C  ➡️  AI 预测明天最高温：**{predicted:.2f}°C**")
    else:
        st.warning("数据库里没有足够的数据来进行 AI 预测。")

except Exception as e:
    # 如果出了任何错，直接把错误打印在网页上，方便我们排错
    st.error(f"哎呀，页面加载出错了：{e}")