import streamlit as st
import pandas as pd
import google.generativeai as genai
import jieba
from collections import Counter
import io

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Market Insight Miner",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stDataFrame {font-size: 14px;}
    [data-testid="stSidebar"] {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 側邊欄設定
# ==========================================
with st.sidebar:
    st.title("💎 Market Miner")
    st.markdown("---")
    
    # 安全輸入 Key
    api_key = st.text_input("請輸入 Gemini API Key", type="password", help="您的 Key 不會被儲存，僅用於本次運算")
    
    if not api_key:
        st.warning("⚠️ 請輸入金鑰以啟動")
        st.stop()
    else:
        try:
            genai.configure(api_key=api_key)
            st.success("✅ AI 連線成功")
        except Exception as e:
            st.error(f"金鑰錯誤: {e}")
            st.stop()

    st.markdown("---")
    mode = st.radio("功能選擇：", ["🌱 模式一：種子關鍵字生成", "⛏️ 模式二：數據挖掘分析"])
    st.markdown("---")
    st.caption("v5.0 Streamlit Edition")

# ==========================================
# 3. 核心函數
# ==========================================
def call_gemini(prompt):
    models = ['gemini-3.0-pro', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-pro']
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            return model.generate_content(prompt).text
        except: continue
    return "❌ 系統忙碌，請稍後再試。"

def clean_google_ads_data(df):
    df.columns = df.columns.str.strip()
    
    # 搜尋量
    search_col = next((c for c in df.columns if 'search' in c.lower() or '搜尋' in c), None)
    if search_col:
        def clean_s(val):
            if pd.isna(val): return 0
            s = str(val).replace(',', '').replace('<', '').replace('>', '').strip()
            if '-' in s:
                try: return (float(s.split('-')[0]) + float(s.split('-')[1])) / 2
                except: return 0
            try: return float(s)
            except: return 0
        df['Avg. monthly searches'] = df[search_col].apply(clean_s)
    else: df['Avg. monthly searches'] = 0

    # 成長率
    yoy_col = next((c for c in df.columns if 'yoy' in c.lower() or 'change' in c.lower() or '變化' in c), None)
    if yoy_col:
        def clean_g(val):
            if pd.isna(val): return 0
            s = str(val).replace('%', '').replace(',', '').replace('+', '').strip()
            if '∞' in s: return 10000
            if '--' in s: return 0
            try: return float(s)
            except: return 0
        df['YoY change'] = df[yoy_col].apply(clean_g)
    else: df['YoY change'] = 0
    
    # 紅海 (High Bid)
    cpc_col = next((c for c in df.columns if ('high' in c.lower() and 'bid' in c.lower()) or '高位' in c), None)
    if cpc_col:
        def clean_price(val):
            if pd.isna(val): return 0
            s = str(val).replace(',', '').replace('NT$', '').replace('$', '').strip()
            if '--' in s: return 0
            try: return float(s)
            except: return 0
        df['Top Page Bid (High)'] = df[cpc_col].apply(clean_price)
    else: df['Top Page Bid (High)'] = 0

    # 藍海 (Competition)
    comp_col = next((c for c in df.columns if 'index' in c.lower() and 'competition' in c.lower()), None)
    if not comp_col: comp_col = next((c for c in df.columns if '競爭' in c and '索引' in c), None)
    df['Competition Index'] = pd.to_numeric(df[comp_col], errors='coerce').fillna(50) if comp_col else 50
        
    return df

def analyze_nlp(keywords, top_n=20):
    text = " ".join(keywords.astype(str).tolist())
    words = jieba.cut(text)
    stop_words = {'的', '推薦', '與', '在', '是', '有', '和', '了', '及', ' ', '什么', '什麼', '怎麼', '如何', '嗎', '食品', '保健', '價格', '多少', '錢'}
    filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
    return pd.DataFrame(Counter(filtered_words).most_common(top_n), columns=['關鍵詞 (Term)', '頻次 (Freq)'])

# ==========================================
# 4. 介面邏輯
# ==========================================

# --- 模式一 ---
if mode == "🌱 模式一：種子關鍵字生成":
    st.header("🌱 Google Ads 種子關鍵字生成")
    st.info("輸入主題，AI 自動生成 3 組策略關鍵字 (每組 10 個)，突破 Google 限制。")
    
    topic = st.text_input("請輸入產品或主題 (例如：益生菌)", "")
    
    if topic and st.button("🚀 生成策略"):
        with st.spinner("AI 正在規劃搜尋戰術..."):
            prompt = f"""
            使用者主題：「{topic}」。
            請生成 3 組 Google Keyword Planner 專用的種子關鍵字 (每組嚴格限制 10 個)。
            請使用 Markdown 格式輸出，不要有多餘廢話。
            
            格式：
            ### 1. 【市場大盤組】(流量型)
            (10個品類大詞，逗號分隔)
            
            ### 2. 【精準轉化組】(痛點型)
            (10個功效/副作用/問句詞，逗號分隔)
            
            ### 3. 【競品攔截組】(藍海型)
            (10個競品或替代方案詞，逗號分隔)
            """
            result = call_gemini(prompt)
            st.markdown(result)
            st.success("請複製上方其中一組貼入 Google Ads。")

# --- 模式二 ---
elif mode == "⛏️ 模式二：數據挖掘分析":
    st.header("⛏️ Google Ads 數據深度挖掘")
    st.info("上傳 CSV，自動進行 NLP 詞頻分析與五維度拆解。")
    
    uploaded_file = st.file_uploader("上傳 Keyword Planner CSV", type=['csv'])
    
    if uploaded_file:
        try:
            try: df = pd.read_csv(uploaded_file, header=2, encoding='utf-16', sep='\t')
            except:
                try: df = pd.read_csv(uploaded_file, header=2, encoding='utf-8')
                except: df = pd.read_csv(uploaded_file, header=2, encoding='latin1')

            df = clean_google_ads_data(df)
            df['Avg. monthly searches'] = pd.to_numeric(df['Avg. monthly searches']).fillna(0)

            # 計算指標
            top_volume = df.sort_values('Avg. monthly searches', ascending=False).head(10)
            growth_base = df[df['Avg. monthly searches'] > 50]
            top_growth = growth_base.sort_values('YoY change', ascending=False).head(10)
            top_decline = growth_base.sort_values('YoY change', ascending=True).head(10)
            theme_freq = analyze_nlp(df[df['Avg. monthly searches'] > 10]['Keyword'], top_n=15)
            red_ocean = df.sort_values('Top Page Bid (High)', ascending=False).head(10)
            blue_ocean = df[(df['Avg. monthly searches'] > 100) & (df['Competition Index'] < 30)].sort_values('Avg. monthly searches', ascending=False).head(10)

            # 顯示 Tabs
            st.divider()
            t1, t2, t3, t4, t5 = st.tabs(["📈 大盤", "🚀 機會", "📉 風險", "🧠 概念", "⚔️ 紅藍海"])
            
            with t1: st.dataframe(top_volume[['Keyword', 'Avg. monthly searches', 'YoY change']].style.background_gradient(subset=['Avg. monthly searches'], cmap='Greens'), use_container_width=True)
            with t2: st.dataframe(top_growth[['Keyword', 'YoY change', 'Avg. monthly searches']].style.background_gradient(subset=['YoY change'], cmap='Reds'), use_container_width=True)
            with t3: st.dataframe(top_decline[['Keyword', 'YoY change', 'Avg. monthly searches']].style.background_gradient(subset=['YoY change'], cmap='Greys'), use_container_width=True)
            with t4: st.bar_chart(theme_freq.set_index('關鍵詞 (Term)'))
            with t5:
                c1, c2 = st.columns(2)
                with c1: 
                    st.markdown("🔥 **紅海 (高出價)**")
                    st.dataframe(red_ocean[['Keyword', 'Top Page Bid (High)']], use_container_width=True)
                with c2: 
                    st.markdown("💧 **藍海 (低競爭)**")
                    st.dataframe(blue_ocean[['Keyword', 'Competition Index', 'Avg. monthly searches']], use_container_width=True)

            # AI 分析
            st.divider()
            if st.button("🧠 呼叫 Gemini 進行戰略解讀"):
                with st.spinner("AI 顧問正在分析中..."):
                    ctx = f"""
                    1.大盤: {top_volume['Keyword'].tolist()}
                    2.機會: {top_growth['Keyword'].tolist()}
                    3.風險: {top_decline['Keyword'].tolist()}
                    4.概念: {theme_freq.values.tolist()}
                    5.紅海: {red_ocean['Keyword'].tolist()}
                    6.藍海: {blue_ocean['Keyword'].tolist()}
                    """
                    prompt = f"""
                    你是一位商業分析師。請根據這五個維度數據歸納：
                    {ctx}
                    請回答：
                    1. **市場隱藏屬性？** (從NLP概念詞中發現了什麼特徵？)
                    2. **新舊勢力交替？** (什麼規格在崛起 vs 消失？)
                    3. **獲利建議？** (避開紅海，切入哪裡？)
                    """
                    res = call_gemini(prompt)
                    st.markdown("### 🧠 Gemini 挖掘報告")
                    st.markdown(res)

        except Exception as e:
            st.error(f"CSV 解析失敗: {e}")
