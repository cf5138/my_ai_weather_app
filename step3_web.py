import streamlit as st
import sqlite3
import pandas as pd
import torch
import torch.nn as nn

# ==========================================
# 1. 页面基础设置与文字介绍（必须放最前面！）
# ==========================================
st.set_page_config(page_title="我的AI天气预测系统", page_icon="🌤️")

# 这两行负责显示大标题和欢迎语
st.title("🌤️ 株洲天气数据看板与 AI 预测")
st.markdown("""
**欢迎来到我的个人网站！** 
这是一个由 *爬虫* + *数据库* + *PyTorch* + *Web* 组成的全栈项目。
👉 [点击这里访问我的 GitHub 主页](https://github.com/cf5138)
""")
st.divider() 

# ==========================================
# 2. 互动模块：输入名字
# ==========================================
name = st.text_input("请输入你的名字：", placeholder="例如：彭于晏、张三、亚洲最杰出的人材...")

if name:
    st.success(f"你好，**{name}**！欢迎来到我的天气预测系统，祝你今天有个好心情！")

st.divider()

# ==========================================
# 3. 数据展示模块（读取 SQLite 数据库）
# ==========================================
st.subheader("📊 近期温度趋势")

try:
    conn = sqlite3.connect('weather.db')
    df = pd.read_sql_query("SELECT * FROM daily_weather", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    st.line_chart(df)
    
    st.write("**详细数据表：**")
    st.dataframe(df)

except Exception as e:
    st.error(f"读取数据库失败啦，错误信息：{e}")

st.divider()

# ==========================================
# 4. AI 预测模块（PyTorch 模型）
# ==========================================
st.subheader("🤖 AI 预测结果")

try:
    temps = df['max_temp'].tolist()
    if len(temps) > 1:
        X = torch.tensor(temps[:-1], dtype=torch.float32).view(-1, 1)
        Y = torch.tensor(temps[1:], dtype=torch.float32).view(-1, 1)
        
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
        
        today_temp = torch.tensor([[temps[-1]]], dtype=torch.float32)
        predicted = model(today_temp).item()
        
        st.info(f"今天最高温：{temps[-1]}°C  ➡️  AI 预测明天最高温：**{predicted:.2f}°C**")
        st.warning("⚠️ 提示：AI 预测仅供娱乐，出门请以当地天气预报为准。")
    else:
        st.warning("数据太少，无法进行 AI 预测。")
except Exception as e:
    st.error(f"AI 预测模块出错了，错误信息：{e}")
