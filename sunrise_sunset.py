# To run:
# E:/VSCode_Project/rent_project/.venv/Scripts/python.exe -m streamlit run E:/VSCode_Project/quant_research/sunrise_sunset.py
import streamlit as st
import ephem
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心数学与天文学算法类
# ==========================================

class SolarMath:
    """
    包含用于根据日照时间反推经纬度的数学公式。
    """
    
    @staticmethod
    def get_solar_declination_and_eot(date_obj):
        day_of_year = date_obj.timetuple().tm_yday
        B = (360 / 365) * (day_of_year - 81)
        B_rad = math.radians(B)
        
        # 1. 计算均时差 (EOT) 单位：分钟
        eot = 9.87 * math.sin(2 * B_rad) - 7.53 * math.cos(B_rad) - 1.5 * math.sin(B_rad)
        
        # 2. 计算太阳赤纬 (delta) 单位：弧度
        delta_deg = 23.45 * math.sin(B_rad)
        delta_rad = math.radians(delta_deg)
        
        return delta_rad, eot

    @staticmethod
    def solve_location(target_date, sunrise_time, sunset_time, utc_offset):
        # 将时间转换为当天的秒数
        sr_seconds = sunrise_time.hour * 3600 + sunrise_time.minute * 60 + sunrise_time.second
        ss_seconds = sunset_time.hour * 3600 + sunset_time.minute * 60 + sunset_time.second
        
        # 计算昼长
        day_length_seconds = ss_seconds - sr_seconds
        if day_length_seconds <= 0:
            return None, "错误：日落时间必须晚于日出时间"
        
        local_solar_noon_seconds = sr_seconds + day_length_seconds / 2
        local_solar_noon_min = local_solar_noon_seconds / 60.0
        
        delta_rad, eot_min = SolarMath.get_solar_declination_and_eot(target_date)
        
        # --- 计算经度 ---
        # 12:00 * 60 = (UTC_noon_min + Longitude_time_offset) + EOT
        utc_noon_min = local_solar_noon_min - (utc_offset * 60)
        long_offset_min = 720 - utc_noon_min - eot_min
        longitude = long_offset_min / 4.0
        
        # --- 计算纬度 ---
        day_length_hours = day_length_seconds / 3600.0
        omega_deg = (day_length_hours / 2) * 15
        omega_rad = math.radians(omega_deg)
        
        tan_delta = math.tan(delta_rad)
        
        if abs(tan_delta) < 0.001:
            return (0.0, longitude), "警告：接近春秋分，纬度计算可能不准确（默认为赤道附近）"
        
        tan_phi = -math.cos(omega_rad) / tan_delta
        phi_rad = math.atan(tan_phi)
        latitude = math.degrees(phi_rad)
        
        return (latitude, longitude), None

# ==========================================
# 2. Ephem 计算引擎
# ==========================================

def calculate_schedule(lat, lon, start_date, days=30):
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 0
    
    sun = ephem.Sun()
    data = []
    
    current_date = start_date
    for _ in range(days):
        observer.date = current_date
        try:
            next_rising = observer.next_rising(sun)
            next_setting = observer.next_setting(sun)
            
            rise_utc = next_rising.datetime()
            set_utc = next_setting.datetime()
            day_len = set_utc - rise_utc
            
            data.append({
                "日期": current_date, 
                "日出UTC": rise_utc,
                "日落UTC": set_utc,
                "昼长": day_len
            })
        except (ephem.AlwaysUpError, ephem.AlwaysDownError):
            pass 
            
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(data)
    if not df.empty:
        df["日期"] = pd.to_datetime(df["日期"])
        
    return df

# ==========================================
# 3. Streamlit 界面逻辑
# ==========================================

st.set_page_config(page_title="太阳反向定位系统", layout="wide")

st.title("☀️ 太阳反向定位与预测系统 Pro")

# --- 初始化 Session State ---
if 'has_calculated' not in st.session_state:
    st.session_state.has_calculated = False

# --- 辅助函数：自定义高精度时间输入组件 ---
def ui_time_input_precise(label, default_h, default_m, default_s, key_prefix):
    st.write(f"**{label}**")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        h = st.number_input(f"时", min_value=0, max_value=23, value=default_h, key=f"{key_prefix}_h", label_visibility="collapsed")
        st.caption("时")
    with c2:
        m = st.number_input(f"分", min_value=0, max_value=59, value=default_m, key=f"{key_prefix}_m", label_visibility="collapsed")
        st.caption("分")
    with c3:
        s = st.number_input(f"秒", min_value=0, max_value=59, value=default_s, key=f"{key_prefix}_s", label_visibility="collapsed")
        st.caption("秒")
    return time(h, m, s)

# --- 侧边栏：输入区域 ---
with st.sidebar:
    st.header("1. 输入观测数据")
    
    with st.form("input_form"):
        input_date = st.date_input("观测日期", datetime.now())
        st.divider()
        
        input_sunrise = ui_time_input_precise("日出时间", 6, 30, 0, "rise")
        st.divider()
        input_sunset = ui_time_input_precise("日落时间", 18, 30, 0, "set")
        st.divider()
            
        utc_offset = st.number_input("所在时区 (UTC偏移)", min_value=-12.0, max_value=14.0, value=8.0, step=0.5)

        st.write("") 
        submitted = st.form_submit_button("计算经纬度 & 生成图表", type="primary", use_container_width=True)
        
        if submitted:
            st.session_state.has_calculated = True

# --- 主界面 ---

if st.session_state.has_calculated:
    # --- 数学原理说明 ---
    with st.expander("📐 点击查看推算背后的数学依据 (Mathematical Logic)", expanded=False):
        st.markdown("本次计算利用了天文学中的 **均时差 (Equation of Time)** 和 **日出方程 (Sunrise Equation)**。")
        math_col1, math_col2 = st.columns(2)
        with math_col1:
            st.markdown("#### 1. 经度 (Longitude)")
            st.latex(r'''
            \begin{aligned}
            T_{noon} &= T_{rise} + \frac{T_{set} - T_{rise}}{2} \\
            \text{Longitude} &= \frac{12:00 - (T_{noon} - \text{Offset} + EOT)}{4 \text{ min}/^{\circ}}
            \end{aligned}
            ''')
        with math_col2:
            st.markdown("#### 2. 纬度 (Latitude)")
            st.latex(r'''
            \phi = \arctan\left( -\frac{\cos(\omega)}{\tan(\delta)} \right)
            ''')
            
    st.divider()

    with st.spinner("正在解算天球几何..."):
        result, error_msg = SolarMath.solve_location(input_date, input_sunrise, input_sunset, utc_offset)
        
        if error_msg and result is None:
            st.error(error_msg)
        else:
            lat, lon = result
            st.success("计算完成！")
            
            # --- 第一部分：反推结果展示 ---
            col_map, col_data = st.columns([3, 2])
            
            with col_data:
                st.subheader("📍 推算位置")
                st.metric("估算纬度", f"{lat:.4f}°")
                st.metric("估算经度", f"{lon:.4f}°")
                if isinstance(error_msg, str):
                    st.warning(error_msg)
                
                day_len_seconds = (input_sunset.hour * 3600 + input_sunset.minute * 60 + input_sunset.second) - \
                                  (input_sunrise.hour * 3600 + input_sunrise.minute * 60 + input_sunrise.second)
                st.caption(f"输入日照时长: {day_len_seconds} 秒")

            with col_map:
                m = folium.Map(location=[lat, lon], zoom_start=6)
                folium.Marker([lat, lon], popup="推算位置", icon=folium.Icon(color="red", icon="sun-o", prefix="fa")).add_to(m)
                st_folium(m, width=500, height=350)
            
            st.divider()
            
            # --- 第二部分：正推与可视化 ---
            st.header("📈 趋势分析与图表")
            
            c1, c2 = st.columns(2)
            with c1:
                calc_start_date = st.date_input("开始日期", input_date)
            with c2:
                days_to_calc = st.number_input("预测天数", 1, 365, 60) 
            
            # 1. 计算基础数据
            schedule_df = calculate_schedule(lat, lon, calc_start_date, days_to_calc)
            
            # 2. 数据处理：转换时区并格式化
            offset_delta = timedelta(hours=utc_offset)
            
            # 为 Plotly 准备数据
            # 使用列表推导式直接生成本地时间列
            schedule_df["LocalRise"] = schedule_df["日出UTC"] + offset_delta
            schedule_df["LocalSet"] = schedule_df["日落UTC"] + offset_delta
            
            # 为了在Y轴上只比较时间（忽略日期的影响），我们创建一个 dummy 时间列
            # 统一把日期设为 2000-01-01，只保留时分秒差异
            def to_dummy_datetime(dt):
                return datetime(2000, 1, 1, dt.hour, dt.minute, dt.second)
            
            schedule_df["DummyRise"] = schedule_df["LocalRise"].apply(to_dummy_datetime)
            schedule_df["DummySet"] = schedule_df["LocalSet"].apply(to_dummy_datetime)
            
            # 统计极值
            local_rises = schedule_df["LocalRise"]
            local_sets = schedule_df["LocalSet"]
            
            earliest_rise_idx = local_rises.apply(lambda x: x.time()).idxmin()
            latest_rise_idx = local_rises.apply(lambda x: x.time()).idxmax()
            earliest_set_idx = local_sets.apply(lambda x: x.time()).idxmin()
            latest_set_idx = local_sets.apply(lambda x: x.time()).idxmax()
            
            st.subheader("📊 关键时间节点")
            k1, k2, k3, k4 = st.columns(4)
            
            with k1:
                r_date = schedule_df.iloc[earliest_rise_idx]["日期"].strftime("%m-%d")
                r_time = local_rises.iloc[earliest_rise_idx].strftime("%H:%M:%S")
                st.metric("最早日出", r_time, delta=f"日期: {r_date}", delta_color="inverse")
                
            with k2:
                r_date = schedule_df.iloc[latest_rise_idx]["日期"].strftime("%m-%d")
                r_time = local_rises.iloc[latest_rise_idx].strftime("%H:%M:%S")
                st.metric("最晚日出", r_time, delta=f"日期: {r_date}", delta_color="inverse")

            with k3:
                s_date = schedule_df.iloc[earliest_set_idx]["日期"].strftime("%m-%d")
                s_time = local_sets.iloc[earliest_set_idx].strftime("%H:%M:%S")
                st.metric("最早日落", s_time, delta=f"日期: {s_date}", delta_color="off")
                
            with k4:
                s_date = schedule_df.iloc[latest_set_idx]["日期"].strftime("%m-%d")
                s_time = local_sets.iloc[latest_set_idx].strftime("%H:%M:%S")
                st.metric("最晚日落", s_time, delta=f"日期: {s_date}")

            # --- 可视化图表 (Plotly Chart) ---
            st.subheader("📉 日出日落趋势图 (双Y轴独立)")
            
            # 创建双 Y 轴图表对象
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 添加日出线 (左轴)
            fig.add_trace(
                go.Scatter(
                    x=schedule_df["日期"], 
                    y=schedule_df["DummyRise"], 
                    name="日出时间",
                    mode='lines+markers',
                    line=dict(color='#FFA500', width=2), # 橙色
                    hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>日出</b>: %{y|%H:%M:%S}<extra></extra>' # 自定义悬停显示
                ),
                secondary_y=False,
            )

            # 添加日落线 (右轴)
            fig.add_trace(
                go.Scatter(
                    x=schedule_df["日期"], 
                    y=schedule_df["DummySet"], 
                    name="日落时间",
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=2), # 蓝色
                    hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>日落</b>: %{y|%H:%M:%S}<extra></extra>'
                ),
                secondary_y=True,
            )

            # 设置布局
            fig.update_layout(
                height=500,
                hovermode="x unified", # 关键：开启X轴统一悬停（会出现纵向虚线，同时显示两个数据）
                xaxis=dict(
                    title="日期",
                    tickformat="%Y-%m-%d",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=20, t=50, b=20)
            )

            # 设置 Y 轴格式 (只显示时:分)
            fig.update_yaxes(
                title_text="日出时间", 
                tickformat="%H:%M", 
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)',
                secondary_y=False
            )
            fig.update_yaxes(
                title_text="日落时间", 
                tickformat="%H:%M", 
                showgrid=False, # 右轴网格线关掉，避免太乱
                secondary_y=True
            )

            # 渲染图表
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 详细数据表格 ---
            with st.expander("查看详细数据表"):
                display_table = pd.DataFrame({
                    "日期": schedule_df["日期"].dt.strftime("%Y-%m-%d"),
                    f"日出 (UTC{utc_offset:+.1f})": schedule_df["LocalRise"].dt.strftime("%H:%M:%S"),
                    f"日落 (UTC{utc_offset:+.1f})": schedule_df["LocalSet"].dt.strftime("%H:%M:%S"),
                    "昼长": schedule_df["昼长"].astype(str).str.split('.').str[0]
                })
                
                st.dataframe(display_table, use_container_width=True, hide_index=True)
                
                csv = display_table.to_csv(index=False).encode('utf-8')
                st.download_button("下载 CSV 数据表", csv, "solar_data.csv", "text/csv")

else:
    st.info("👈 请在左侧侧边栏输入观测数据并点击“计算经纬度”")