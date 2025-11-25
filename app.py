import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from analisys import LinearModel, QuadraticModel, ExponentialModel

st.set_page_config(page_title='Auto-Fitter for Lab', layout='wide')
st.title('Auto-Fitter for Lab')
st.write('実験データをアップロードすると、最適なモデルを自動選定します。')
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
        run_btn = st.button('解析開始', type='primary')
    
    if run_btn:
        x_data = df[x_col]
        y_data = df[y_col]

        models = [LinearModel(), QuadraticModel(), ExponentialModel()]
        results = []

        fig, ax = plt.subplots()
        ax.scatter(x_data, y_data, label='raw data', color='black')
        x_range = np.linspace(min(x_data), max(x_data), 100)

        best_aic = np.inf
        best_model = None

        for model in models:
            model.fit(x_data, y_data)
            aic = model.get_aic()
            results.append({
                'Model': model.__class__.__name__,
                'AIC': round(aic, 2),
                'Equation': model.get_equation(),        
                'Params': np.round(model.params, 3)
            })
            if aic < best_aic:
                best_aic = aic
                best_model = model

        if best_model is not None:
            x_range = np.linspace(min(x_data), max(x_data), 200)
            y_pred_plot = best_model.func(x_range, *best_model.params)
            ax.plot(x_range, y_pred_plot, label=f"Best: {best_model.__class__.__name__}")

        st.header('モデル評価結果')
        df_results = pd.DataFrame(results).sort_values(by="AIC")
        st.dataframe(df_results)
        st.info('AICが低いモデルほどデータの当てはまりとシンプルさのバランスが良いと判断されます。')



                    



