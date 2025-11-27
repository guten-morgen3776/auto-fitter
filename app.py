import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from analisys import (
    LinearModel, QuadraticModel, ExponentialModel, 
    LogarithmicModel, SigmoidModel, MichaelisMentenModel
)
import google.generativeai as genai
import os

if 'GEMINI_API_KEY' in st.secrets:
    genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
else:
    st.error('APIキーが設定されていません。')

st.set_page_config(
    page_title='Auto-Fitter for Lab', 
    layout='wide'
)
st.title('Auto-Fitter for Lab')
st.write('データをアップロードすると、最適なモデルを自動選定します。')
st.markdown('データの入力方法を選んでください。')

tab1, tab2 = st.tabs(['CSVファイルをアップロード','直接入力'])
df = None
with tab1:
    uploaded_file = st.file_uploader('CSVファイルをドラッグ&ドロップ', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
with tab2:
    st.info('以下の表にデータを直接入力してください。行の追加も可能です。')
    default_data = pd.DataFrame({
        "x": [0.0], "y": [0.0]
    })
    edited_df = st.data_editor(default_data, num_rows='dynamic')
    if not edited_df.empty and df is None:
        df = edited_df

def get_ai_insight(df_results, user_context):
    results_text = df_results.to_markdown(index=False)
    best_model = df_results.iloc[0]
    best_model_name = best_model['Model']
    best_eq = best_model['Equation']

    prompt = f"""
    あなたは一流の自然科学の専門家です。
    以下のデータ解析結果と、ユーザーから提供されたデータの背景情報に基づき、
    学術的かつ論理的な考察を生成してください。

    ### 解析結果（AICによるモデル比較）
    {results_text}

    ### 最も当てはまりが良かったモデル
    モデル名: {best_model_name}
    数式: {best_eq}

    ### データの背景
    {user_context}

    ### 指示
    1. ベストモデルが選ばれた理由をデータの形状やAICの観点から簡潔に述べてください。
    2. その数式やパラメータ値が背景情報と照らし合わせて、科学的に何を意味する考察してください。
    3. もしモデルの当てはまりが悪い、または不確かさが大きい場合はその原因として考えられること（測定誤差、モデルの不備など）を指摘してください。
    4. レポートにそのまま使えるような、「だ・である」調で出力してください。
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f'考察の生成に失敗しました: {e}'

if df is not None:
    st.write('データのプレビュー')
    st.dataframe(df.head())
    st.divider()

    cols = df.columns.tolist()
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        x_col = st.selectbox('X軸のデータを選んでください', cols, index=0)
    with c2:
        y_col = st.selectbox('Y軸のデータを選んでください', cols, index=1 if len(cols)>1 else 0)
    with c3:
        st.write('')
        st.write('')
        run_btn = st.button('解析開始', type='primary')
    
    if run_btn:
        x_data = df[x_col]
        y_data = df[y_col]

        models = [
            LinearModel(), 
            QuadraticModel(), 
            ExponentialModel(),
            LogarithmicModel(),
            SigmoidModel(),
            MichaelisMentenModel()
        ]
        results = []

        fig, ax = plt.subplots()
        ax.scatter(x_data, y_data, label='raw data', color='black')
        x_range = np.linspace(min(x_data), max(x_data), 100)

        best_aicc = np.inf
        best_model = None

        for model in models:
            model.fit(x_data, y_data)
            aicc = model.get_aicc()
            results.append({
                'Model': model.__class__.__name__,
                'AICc': round(aicc, 2),
                'Equation': model.get_equation(),        
                'Params': np.round(model.params, 3)
            })
            if aicc < best_aicc:
                best_aicc = aicc
                best_model = model

        if best_model is not None:
            st.session_state['analysis_result'] = pd.DataFrame(results).sort_values(by='AICc')
            st.session_state['best_model'] = best_model
            st.session_state['x_data'] = x_data
            st.session_state['y_data'] = y_data
            st.session_state['x_col'] = x_col
            st.session_state['y_col'] = y_col
    if 'analysis_result' in st.session_state:
        df_results = st.session_state['analysis_result']
        best_model = st.session_state['best_model']
        x_data = st.session_state['x_data']
        y_data = st.session_state['y_data']
        x_col_name = st.session_state['x_col']
        y_col_name = st.session_state['y_col']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_data, y_data, label='Raw Data', color='black')
        x_range = np.linspace(min(x_data), max(x_data), 200)
        y_pred_plot = best_model.func(x_range, *best_model.params)

        lower, upper = best_model.get_confidence_interval(x_range)
        if lower is not None:
            ax.fill_between(x_range, lower, upper, color='red', alpha=0.2, label='95% Confidence Interval')
            
        ax.plot(x_range, y_pred_plot, label=f"Best: {best_model.__class__.__name__}", color='red')
        ax.legend()
        ax.set_title(f"Best Fit: {best_model.__class__.__name__}")
        ax.set_xlabel(x_col_name)
        ax.set_ylabel(y_col_name)

        col_res1, col_res2 = st.columns([3, 2])
        with col_res1:
            st.subheader("Fitting Plot")
            st.pyplot(fig) 
        with col_res2:
            st.subheader('Model Ranking')
            st.dataframe(df_results)
            st.info('AICcが低いモデルほどデータの当てはまりとシンプルさのバランスが良いと判断されます。')



st.divider()
st.header('AIアシスタントによる考察')
st.write('データの背景や条件を入力すると、解析結果に基づいた考察案を作成します。')
user_context = st.text_area(
    'データの背景（例：これは酵素キモトリプシンの反応速度データです。基質濃度と反応速度の関係を見ています。）',
    height=100
)
if st.button('AIに考察を書かせる。'):
    if 'analysis_result' not in st.session_state:
        st.error('まずは「解析開始」ボタンを押して、データを分析してください。')
    elif not user_context:
        st.warning('考察の精度を高めるために実験の背景を入力してください。')
    else:
        with st.spinner('AIがデータを読み込んでいます...'):
            df_results = st.session_state['analysis_result']
            insight = get_ai_insight(df_results, user_context)
            st.markdown('AI考察案')
            st.success('考察が生成されました！')
            st.markdown(insight)



                    



