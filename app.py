import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import jieba
from collections import Counter
from itertools import combinations
import io
import json
import re

# ==========================================
# 0. 全域設定
# ==========================================
if 'usage_count' not in st.session_state:
    st.session_state['usage_count'] = 0

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Market Insight Miner v6.0",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stDataFrame {font-size: 14px;}
    [data-testid="stSidebar"] {background-color: #f0f2f6;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 側邊欄
# ==========================================
with st.sidebar:
    st.title("💎 Market Miner v6")
    st.caption("詞彙結構 × 商業價值分析")
    st.markdown("---")
    
    api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        help="請輸入 Google AI Studio 提供的 API Key"
    )
    st.caption("[取得 API Key](https://aistudio.google.com/app/apikey)")
    
    if api_key:
        st.success("✅ API Key 已設定")
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"API 設定失敗: {e}")
    else:
        st.warning("⚠️ 請輸入 Key 啟用 AI")

    st.markdown("---")
    st.subheader("🧠 模型選擇")
    selected_model = st.selectbox(
        "Gemini 模型",
        [
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        index=0,
        help="Flash 較快較便宜，Pro 較精準"
    )
    st.session_state['selected_model'] = selected_model
    
    st.markdown("---")
    st.metric("本次使用次數", st.session_state['usage_count'])
    
    st.markdown("---")
    mode = st.radio(
        "功能選擇", 
        [
            "🌱 種子關鍵字生成",
            "⛏️ 詞彙結構分析",
            "🔗 詞彙關聯探勘"
        ]
    )
    st.markdown("---")
    st.caption("v6.0 | 詞彙深度分析版")

# ==========================================
# 3. 核心函數
# ==========================================
def call_gemini(prompt, model_name=None):
    """呼叫 Gemini API"""
    if not api_key:
        return "⚠️ 請先輸入 API Key"
    
    # 使用傳入的模型或 sidebar 選擇的模型
    use_model = model_name if model_name else st.session_state.get('selected_model', 'gemini-2.0-flash')
    
    try:
        model = genai.GenerativeModel(use_model)
        response = model.generate_content(prompt)
        st.session_state['usage_count'] += 1
        return response.text
    except Exception as e:
        return f"❌ AI 呼叫失敗: {str(e)}"


def clean_google_ads_data(df):
    """清理 Google Ads CSV 資料"""
    df.columns = df.columns.str.strip()
    
    # 關鍵字欄位
    kw_col = next((c for c in df.columns if 'keyword' in c.lower() or '關鍵字' in c), None)
    if kw_col and kw_col != 'Keyword':
        df['Keyword'] = df[kw_col]
    
    # 搜尋量
    search_col = next((c for c in df.columns if 'search' in c.lower() or '搜尋' in c), None)
    if search_col:
        def clean_search(val):
            if pd.isna(val): return 0
            s = str(val).replace(',', '').replace('<', '').replace('>', '').strip()
            if '-' in s:
                try: 
                    parts = s.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                except: return 0
            try: return float(s)
            except: return 0
        df['Avg. monthly searches'] = df[search_col].apply(clean_search)
    else: 
        df['Avg. monthly searches'] = 0

    # YoY 成長率
    yoy_col = next((c for c in df.columns if 'yoy' in c.lower() or 'change' in c.lower() or '變化' in c), None)
    if yoy_col:
        def clean_yoy(val):
            if pd.isna(val): return 0
            s = str(val).replace('%', '').replace(',', '').replace('+', '').strip()
            if '∞' in s: return 999
            if '--' in s or s == '': return 0
            try: return float(s)
            except: return 0
        df['YoY change'] = df[yoy_col].apply(clean_yoy)
    else: 
        df['YoY change'] = 0
    
    # High Bid (紅海指標)
    cpc_col = next((c for c in df.columns if ('high' in c.lower() and 'bid' in c.lower()) or '高位' in c), None)
    if cpc_col:
        def clean_bid(val):
            if pd.isna(val): return 0
            s = str(val).replace(',', '').replace('NT$', '').replace('$', '').strip()
            if '--' in s or s == '': return 0
            try: return float(s)
            except: return 0
        df['Top Page Bid (High)'] = df[cpc_col].apply(clean_bid)
    else: 
        df['Top Page Bid (High)'] = 0

    # Competition Index (藍海指標)
    comp_col = next((c for c in df.columns if 'index' in c.lower() and 'competition' in c.lower()), None)
    if not comp_col:
        comp_col = next((c for c in df.columns if '競爭' in c and '索引' in c), None)
    if comp_col:
        df['Competition Index'] = pd.to_numeric(df[comp_col], errors='coerce').fillna(50)
    else:
        df['Competition Index'] = 50
        
    return df


def tokenize_keywords(keywords_series, stop_words=None):
    """對關鍵字進行分詞"""
    if stop_words is None:
        stop_words = {
            '的', '推薦', '與', '在', '是', '有', '和', '了', '及', ' ', 
            '什么', '什麼', '怎麼', '如何', '嗎', '價格', '多少', '錢',
            'ptt', 'dcard', '哪裡', '可以', '要', '會', '能', '好'
        }
    
    all_tokens = []
    for kw in keywords_series.astype(str):
        tokens = list(jieba.cut(kw))
        filtered = [t for t in tokens if len(t) > 1 and t not in stop_words]
        all_tokens.extend(filtered)
    
    return all_tokens


def analyze_word_frequency(keywords_series, top_n=20):
    """詞頻分析"""
    tokens = tokenize_keywords(keywords_series)
    freq = Counter(tokens).most_common(top_n)
    return pd.DataFrame(freq, columns=['詞彙', '頻次'])


def analyze_cooccurrence(keywords_series, top_n=30):
    """
    共現分析：找出經常一起出現的詞組
    """
    stop_words = {
        '的', '推薦', '與', '在', '是', '有', '和', '了', '及', ' ',
        '什么', '什麼', '怎麼', '如何', '嗎', '價格', '多少', '錢',
        'ptt', 'dcard', '哪裡', '可以', '要', '會', '能', '好'
    }
    
    cooccur_counter = Counter()
    
    for kw in keywords_series.astype(str):
        tokens = list(jieba.cut(kw))
        filtered = [t for t in tokens if len(t) > 1 and t not in stop_words]
        # 取所有兩兩組合
        for pair in combinations(sorted(set(filtered)), 2):
            cooccur_counter[pair] += 1
    
    # 轉成 DataFrame
    data = [(p[0], p[1], c) for p, c in cooccur_counter.most_common(top_n)]
    return pd.DataFrame(data, columns=['詞彙A', '詞彙B', '共現次數'])


def analyze_ngrams(keywords_series, n=2, top_k=20):
    """
    N-gram 分析：找出常見的連續詞組
    """
    ngram_counter = Counter()
    
    for kw in keywords_series.astype(str):
        tokens = list(jieba.cut(kw))
        tokens = [t for t in tokens if len(t.strip()) > 0]
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            # 過濾太短或無意義的
            if all(len(t) > 1 for t in ngram):
                ngram_counter[ngram] += 1
    
    data = [(' '.join(ng), c) for ng, c in ngram_counter.most_common(top_k)]
    return pd.DataFrame(data, columns=[f'{n}-gram 詞組', '出現次數'])


def calculate_word_value(df, keywords_series):
    """
    計算詞彙商業價值
    結合：出現頻次 × 平均搜尋量 × 平均出價
    """
    stop_words = {
        '的', '推薦', '與', '在', '是', '有', '和', '了', '及', ' ',
        '什么', '什麼', '怎麼', '如何', '嗎', '價格', '多少', '錢',
        'ptt', 'dcard', '哪裡', '可以', '要', '會', '能', '好'
    }
    
    word_stats = {}
    
    for idx, kw in enumerate(keywords_series.astype(str)):
        tokens = list(jieba.cut(kw))
        filtered = [t for t in tokens if len(t) > 1 and t not in stop_words]
        
        row = df.iloc[idx] if idx < len(df) else None
        if row is None:
            continue
            
        search_vol = row.get('Avg. monthly searches', 0)
        bid = row.get('Top Page Bid (High)', 0)
        yoy = row.get('YoY change', 0)
        
        for token in filtered:
            if token not in word_stats:
                word_stats[token] = {
                    'count': 0,
                    'total_search': 0,
                    'total_bid': 0,
                    'total_yoy': 0
                }
            word_stats[token]['count'] += 1
            word_stats[token]['total_search'] += search_vol
            word_stats[token]['total_bid'] += bid
            word_stats[token]['total_yoy'] += yoy
    
    # 計算指標
    results = []
    for word, stats in word_stats.items():
        count = stats['count']
        avg_search = stats['total_search'] / count if count > 0 else 0
        avg_bid = stats['total_bid'] / count if count > 0 else 0
        avg_yoy = stats['total_yoy'] / count if count > 0 else 0
        
        # 商業價值分數 = 頻次權重 × 搜尋量權重 × 出價權重
        value_score = (
            np.log1p(count) * 0.3 +
            np.log1p(avg_search) * 0.4 +
            np.log1p(avg_bid) * 0.3
        ) * 10
        
        results.append({
            '詞彙': word,
            '出現次數': count,
            '平均搜尋量': round(avg_search, 0),
            '平均出價': round(avg_bid, 1),
            '平均YoY': round(avg_yoy, 1),
            '商業價值分': round(value_score, 2)
        })
    
    result_df = pd.DataFrame(results)
    return result_df.sort_values('商業價值分', ascending=False)


def analyze_word_trends(df, keywords_series):
    """
    詞彙趨勢分群：上升詞、下降詞、穩定詞
    """
    stop_words = {
        '的', '推薦', '與', '在', '是', '有', '和', '了', '及', ' ',
        '什么', '什麼', '怎麼', '如何', '嗎', '價格', '多少', '錢',
        'ptt', 'dcard', '哪裡', '可以', '要', '會', '能', '好'
    }
    
    word_yoy = {}
    
    for idx, kw in enumerate(keywords_series.astype(str)):
        tokens = list(jieba.cut(kw))
        filtered = [t for t in tokens if len(t) > 1 and t not in stop_words]
        
        row = df.iloc[idx] if idx < len(df) else None
        if row is None:
            continue
            
        yoy = row.get('YoY change', 0)
        search_vol = row.get('Avg. monthly searches', 0)
        
        for token in filtered:
            if token not in word_yoy:
                word_yoy[token] = {'yoy_values': [], 'search_values': []}
            word_yoy[token]['yoy_values'].append(yoy)
            word_yoy[token]['search_values'].append(search_vol)
    
    results = []
    for word, data in word_yoy.items():
        avg_yoy = np.mean(data['yoy_values'])
        avg_search = np.mean(data['search_values'])
        count = len(data['yoy_values'])
        
        # 分群
        if avg_yoy > 20:
            trend = '🚀 上升'
        elif avg_yoy < -20:
            trend = '📉 下降'
        else:
            trend = '➡️ 穩定'
        
        results.append({
            '詞彙': word,
            '平均YoY': round(avg_yoy, 1),
            '平均搜尋量': round(avg_search, 0),
            '出現次數': count,
            '趨勢': trend
        })
    
    return pd.DataFrame(results).sort_values('平均YoY', ascending=False)


def parse_intent_data(uploaded_json):
    """解析 SERP 雷達的意圖研究結果"""
    try:
        if isinstance(uploaded_json, str):
            data = json.loads(uploaded_json)
        else:
            data = json.load(uploaded_json)
        
        # 支援陣列或單一物件
        if isinstance(data, list):
            return data
        return [data]
    except Exception as e:
        return None


# ==========================================
# 4. 模式一：種子關鍵字生成
# ==========================================
if mode == "🌱 種子關鍵字生成":
    st.header("🌱 Google Ads 種子關鍵字生成")
    st.info("輸入主題，AI 生成 3 組策略關鍵字")
    
    topic = st.text_input("產品或主題", placeholder="例如：益生菌、空氣清淨機")
    
    if topic and st.button("🚀 生成策略", type="primary"):
        if not api_key:
            st.error("請先輸入 API Key")
        else:
            with st.spinner("AI 規劃中..."):
                prompt = f"""
                主題：「{topic}」
                
                請生成 3 組 Google Keyword Planner 種子關鍵字（每組 10 個）。
                
                格式要求（Markdown）：
                
                ### 1. 【市場大盤組】流量型
                (10個品類大詞，逗號分隔)
                
                ### 2. 【精準轉化組】痛點型
                (10個功效/問題/問句詞，逗號分隔)
                
                ### 3. 【競品攔截組】藍海型
                (10個競品或替代方案詞，逗號分隔)
                
                直接輸出，不要多餘說明。
                """
                result = call_gemini(prompt)
                st.markdown(result)


# ==========================================
# 5. 模式二：詞彙結構分析
# ==========================================
elif mode == "⛏️ 詞彙結構分析":
    st.header("⛏️ 詞彙結構 × 商業價值分析")
    
    # 上傳區域
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.subheader("📊 Google Ads CSV（支援多檔）")
        uploaded_csvs = st.file_uploader(
            "上傳 Keyword Planner CSV",
            type=['csv'],
            key="csv_upload",
            accept_multiple_files=True
        )
    
    with col_upload2:
        st.subheader("🎯 意圖研究結果（選填）")
        intent_input_method = st.radio(
            "輸入方式",
            ["上傳 JSON", "貼上文字"],
            horizontal=True
        )
        
        intent_data = None
        if intent_input_method == "上傳 JSON":
            uploaded_intent = st.file_uploader(
                "上傳 SERP 雷達 JSON",
                type=['json'],
                key="intent_upload"
            )
            if uploaded_intent:
                intent_data = parse_intent_data(uploaded_intent)
                if intent_data:
                    st.success(f"✅ 已載入 {len(intent_data)} 筆意圖資料")
        else:
            intent_text = st.text_area(
                "貼上 SERP 雷達 JSON",
                height=150,
                placeholder='[{"Keyword": "...", "User_Intent": "...", ...}]'
            )
            if intent_text.strip():
                intent_data = parse_intent_data(intent_text)
                if intent_data:
                    st.success(f"✅ 已解析 {len(intent_data)} 筆意圖資料")

    # 處理 CSV（支援多檔合併）
    if uploaded_csvs:
        try:
            all_dfs = []
            file_stats = []
            
            for uploaded_csv in uploaded_csvs:
                # 嘗試多種編碼
                try:
                    single_df = pd.read_csv(uploaded_csv, header=2, encoding='utf-16', sep='\t')
                except:
                    try:
                        single_df = pd.read_csv(uploaded_csv, header=2, encoding='utf-8')
                    except:
                        single_df = pd.read_csv(uploaded_csv, header=2, encoding='latin1')
                
                single_df = clean_google_ads_data(single_df)
                single_df['_source_file'] = uploaded_csv.name  # 標記來源
                all_dfs.append(single_df)
                file_stats.append({
                    'file': uploaded_csv.name,
                    'rows': len(single_df)
                })
            
            # 合併所有檔案
            df = pd.concat(all_dfs, ignore_index=True)
            
            # 去重（同關鍵字保留搜尋量較高的）
            df = df.sort_values('Avg. monthly searches', ascending=False)
            df = df.drop_duplicates(subset=['Keyword'], keep='first')
            
            st.divider()
            
            # 顯示檔案統計
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("上傳檔案數", len(uploaded_csvs))
            with col_stat2:
                st.metric("合併後關鍵字", len(df))
            with col_stat3:
                st.metric("去重前總數", sum(f['rows'] for f in file_stats))
            
            with st.expander("📁 檔案明細"):
                for f in file_stats:
                    st.caption(f"• {f['file']}: {f['rows']} 筆")
            
            # 顯示意圖摘要（如果有）
            if intent_data:
                with st.expander("🎯 意圖研究摘要", expanded=True):
                    for item in intent_data[:5]:  # 最多顯示 5 筆
                        st.markdown(f"**{item.get('Keyword', 'N/A')}**")
                        st.caption(f"意圖：{item.get('User_Intent', 'N/A')}")
                        st.caption(f"機會：{item.get('Opportunity_Gap', 'N/A')}")
            
            # ===== 分析 Tabs =====
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 詞頻分析",
                "💰 商業價值",
                "📈 趨勢分群",
                "🔴🔵 紅藍海",
                "🧠 AI 洞察"
            ])
            
            keywords_col = df['Keyword'] if 'Keyword' in df.columns else df.iloc[:, 0]
            
            # Tab 1: 詞頻
            with tab1:
                st.subheader("詞彙出現頻次")
                freq_df = analyze_word_frequency(keywords_col, top_n=25)
                
                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(freq_df.set_index('詞彙')['頻次'])
                with col_table:
                    st.dataframe(freq_df, use_container_width=True, height=400)
            
            # Tab 2: 商業價值
            with tab2:
                st.subheader("詞彙商業價值排行")
                st.caption("價值分 = 頻次(30%) × 搜尋量(40%) × 出價(30%)")
                
                value_df = calculate_word_value(df, keywords_col)
                
                # 篩選
                min_count = st.slider("最低出現次數", 1, 10, 2)
                filtered_value = value_df[value_df['出現次數'] >= min_count].head(30)
                
                st.dataframe(
                    filtered_value.style.background_gradient(
                        subset=['商業價值分'], 
                        cmap='YlOrRd'
                    ),
                    use_container_width=True,
                    height=500
                )
            
            # Tab 3: 趨勢分群
            with tab3:
                st.subheader("詞彙趨勢分群")
                
                trend_df = analyze_word_trends(df, keywords_col)
                
                col_up, col_down, col_stable = st.columns(3)
                
                rising = trend_df[trend_df['趨勢'] == '🚀 上升'].head(15)
                falling = trend_df[trend_df['趨勢'] == '📉 下降'].head(15)
                stable = trend_df[trend_df['趨勢'] == '➡️ 穩定'].head(15)
                
                with col_up:
                    st.markdown("### 🚀 上升詞")
                    st.dataframe(rising[['詞彙', '平均YoY', '平均搜尋量']], height=400)
                
                with col_down:
                    st.markdown("### 📉 下降詞")
                    st.dataframe(falling[['詞彙', '平均YoY', '平均搜尋量']], height=400)
                
                with col_stable:
                    st.markdown("### ➡️ 穩定詞")
                    st.dataframe(stable[['詞彙', '平均YoY', '平均搜尋量']], height=400)
            
            # Tab 4: 紅藍海
            with tab4:
                st.subheader("紅藍海關鍵字")
                
                col_red, col_blue = st.columns(2)
                
                with col_red:
                    st.markdown("### 🔥 紅海（高競爭高出價）")
                    red_ocean = df.nlargest(15, 'Top Page Bid (High)')
                    st.dataframe(
                        red_ocean[['Keyword', 'Top Page Bid (High)', 'Avg. monthly searches']],
                        use_container_width=True
                    )
                
                with col_blue:
                    st.markdown("### 💧 藍海（低競爭有量）")
                    blue_ocean = df[
                        (df['Avg. monthly searches'] > 100) & 
                        (df['Competition Index'] < 40)
                    ].nlargest(15, 'Avg. monthly searches')
                    st.dataframe(
                        blue_ocean[['Keyword', 'Competition Index', 'Avg. monthly searches']],
                        use_container_width=True
                    )
            
            # Tab 5: AI 洞察
            with tab5:
                st.subheader("🧠 AI 深度洞察")
                
                if st.button("啟動 AI 分析", type="primary"):
                    if not api_key:
                        st.error("請先輸入 API Key")
                    else:
                        with st.spinner("AI 分析中..."):
                            # 準備資料摘要
                            freq_top = analyze_word_frequency(keywords_col, 15)
                            value_top = calculate_word_value(df, keywords_col).head(15)
                            trend_summary = analyze_word_trends(df, keywords_col)
                            
                            rising_words = trend_summary[trend_summary['趨勢'] == '🚀 上升']['詞彙'].head(10).tolist()
                            falling_words = trend_summary[trend_summary['趨勢'] == '📉 下降']['詞彙'].head(10).tolist()
                            
                            # 意圖資料
                            intent_context = ""
                            if intent_data:
                                intent_context = f"""
                                
                                【SERP 意圖研究結果】
                                {json.dumps(intent_data[:5], ensure_ascii=False, indent=2)}
                                """
                            
                            prompt = f"""
                            你是市場研究分析師。請根據以下詞彙結構數據，提供商業洞察。
                            
                            【高頻詞彙】
                            {freq_top.to_string(index=False)}
                            
                            【高價值詞彙】（商業價值分 = 頻次×搜尋量×出價）
                            {value_top[['詞彙', '商業價值分', '平均搜尋量', '平均出價']].to_string(index=False)}
                            
                            【上升趨勢詞】
                            {rising_words}
                            
                            【下降趨勢詞】
                            {falling_words}
                            {intent_context}
                            
                            請分析：
                            
                            ## 1. 市場結構解讀
                            從詞彙頻次看出什麼市場特徵？哪些概念是核心？
                            
                            ## 2. 價值機會點
                            高價值詞彙揭示了什麼商業機會？建議優先攻佔哪些詞？
                            
                            ## 3. 趨勢判讀
                            上升詞代表什麼新興需求？下降詞代表什麼在退場？
                            
                            ## 4. 策略建議
                            基於以上分析，給出 3 個具體可執行的內容/廣告策略。
                            
                            請用繁體中文回答，直接輸出分析，不要重複數據。
                            """
                            
                            result = call_gemini(prompt)
                            st.markdown(result)
            
            # ===== Excel 下載 =====
            st.divider()
            if st.button("📥 匯出完整分析 Excel"):
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # 原始資料
                    df.to_excel(writer, sheet_name='Raw_Data', index=False)
                    
                    # 詞頻
                    analyze_word_frequency(keywords_col, 50).to_excel(
                        writer, sheet_name='Word_Frequency', index=False
                    )
                    
                    # 商業價值
                    calculate_word_value(df, keywords_col).head(100).to_excel(
                        writer, sheet_name='Word_Value', index=False
                    )
                    
                    # 趨勢
                    analyze_word_trends(df, keywords_col).to_excel(
                        writer, sheet_name='Word_Trends', index=False
                    )
                
                st.download_button(
                    label="⬇️ 下載 Excel",
                    data=buffer.getvalue(),
                    file_name=f"market_miner_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        except Exception as e:
            st.error(f"CSV 解析失敗: {e}")
            st.info("請確認上傳的是 Google Keyword Planner 原始 CSV")


# ==========================================
# 6. 模式三：詞彙關聯探勘
# ==========================================
elif mode == "🔗 詞彙關聯探勘":
    st.header("🔗 詞彙關聯 × 共現分析")
    st.info("分析詞與詞之間的關聯性，找出隱藏的語意結構")
    
    uploaded_csvs = st.file_uploader(
        "上傳 Keyword Planner CSV（支援多檔）", 
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_csvs:
        try:
            all_dfs = []
            
            for uploaded_csv in uploaded_csvs:
                try:
                    single_df = pd.read_csv(uploaded_csv, header=2, encoding='utf-16', sep='\t')
                except:
                    try:
                        single_df = pd.read_csv(uploaded_csv, header=2, encoding='utf-8')
                    except:
                        single_df = pd.read_csv(uploaded_csv, header=2, encoding='latin1')
                
                single_df = clean_google_ads_data(single_df)
                all_dfs.append(single_df)
            
            df = pd.concat(all_dfs, ignore_index=True)
            df = df.sort_values('Avg. monthly searches', ascending=False)
            df = df.drop_duplicates(subset=['Keyword'], keep='first')
            
            keywords_col = df['Keyword'] if 'Keyword' in df.columns else df.iloc[:, 0]
            
            st.success(f"✅ 已載入 {len(uploaded_csvs)} 個檔案，合併 {len(df)} 筆關鍵字")
            st.divider()
            
            tab_cooccur, tab_ngram, tab_network = st.tabs([
                "🔗 共現分析",
                "📝 N-gram 詞組",
                "🕸️ 關聯解讀"
            ])
            
            # Tab: 共現
            with tab_cooccur:
                st.subheader("詞彙共現矩陣")
                st.caption("哪些詞經常一起出現？揭示使用者的組合搜尋習慣")
                
                top_n_cooccur = st.slider("顯示前 N 組", 10, 50, 30)
                cooccur_df = analyze_cooccurrence(keywords_col, top_n=top_n_cooccur)
                
                st.dataframe(
                    cooccur_df.style.background_gradient(subset=['共現次數'], cmap='Blues'),
                    use_container_width=True,
                    height=500
                )
            
            # Tab: N-gram
            with tab_ngram:
                st.subheader("N-gram 詞組分析")
                
                col_2gram, col_3gram = st.columns(2)
                
                with col_2gram:
                    st.markdown("### 2-gram（雙詞組）")
                    bigram_df = analyze_ngrams(keywords_col, n=2, top_k=20)
                    st.dataframe(bigram_df, use_container_width=True)
                
                with col_3gram:
                    st.markdown("### 3-gram（三詞組）")
                    trigram_df = analyze_ngrams(keywords_col, n=3, top_k=20)
                    st.dataframe(trigram_df, use_container_width=True)
            
            # Tab: AI 關聯解讀
            with tab_network:
                st.subheader("🧠 AI 關聯結構解讀")
                
                if st.button("啟動關聯分析", type="primary"):
                    if not api_key:
                        st.error("請先輸入 API Key")
                    else:
                        with st.spinner("AI 解讀詞彙關聯中..."):
                            cooccur_data = analyze_cooccurrence(keywords_col, 30)
                            bigram_data = analyze_ngrams(keywords_col, 2, 20)
                            trigram_data = analyze_ngrams(keywords_col, 3, 15)
                            
                            prompt = f"""
                            你是語意分析專家。請解讀以下詞彙關聯數據。
                            
                            【共現詞組】（經常一起出現的詞）
                            {cooccur_data.to_string(index=False)}
                            
                            【2-gram 詞組】
                            {bigram_data.to_string(index=False)}
                            
                            【3-gram 詞組】
                            {trigram_data.to_string(index=False)}
                            
                            請分析：
                            
                            ## 1. 核心概念叢集
                            從共現關係中，識別出 3-5 個核心概念群（哪些詞形成一個主題？）
                            
                            ## 2. 使用者搜尋模式
                            從 N-gram 詞組看出使用者用什麼句式/結構在搜尋？
                            
                            ## 3. 隱藏需求
                            這些關聯揭示了什麼未被明說的使用者需求？
                            
                            ## 4. 內容策略建議
                            基於詞彙關聯，建議製作什麼類型的內容來覆蓋這些詞組？
                            
                            用繁體中文回答。
                            """
                            
                            result = call_gemini(prompt)
                            st.markdown(result)
                            
        except Exception as e:
            st.error(f"CSV 解析失敗: {e}")
