import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analisys import LinearModel, QuadraticModel, ExponentialModel

st.title('Auto-Fitter for Lab')
st.write('実験データをアップロードすると、最適なモデルを自動選定します。')

uploaded_file = st.file_uploader('CSVファイルをドラッグ&ドロップ', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('データのプレビュー')
    st.dataframe(df.head())

    columns = df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox('X軸のデータを選んでください', columns)
    with col2:
        y_col = st.selectbox('Y軸のデータを選んでください', columns)
    
    if st.button('解析開始'):
        x_data = df[x_col]
        y_data = df[y_col]

        models = [LinearModel(), QuadraticModel(), ExponentialModel()]
        results = []

        fig, ax = plt.subplots()
        ax.scatter(x_data, y_data, label='raw data', color='black')
        x_range = np.linspace(min(x_data), max(x_data), 100)

        best_aic = np.inf
        best_model_name = ''

        for model in models:
            model.fit(x_data, y_data)
            aic = model.get_aic()
            results.append({
                'Model': model.__class__.__name__,
                'AIC': round(aic, 2),
                'Equation': model.get_equation(),        
                'Params': np.round(model.params, 3)
            })

            y_pred_plot = model.func(x_range, *model.params)
            ax.plot(x_range, y_pred_plot, label=model.__class__.__name__)

            if aic < best_aic:
                best_aic = aic
                best_model_name = model.__class__.__name__

        ax.legend()
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'best fit: {best_model_name}')
        st.pyplot(fig)

        st.header('モデル評価結果')
        df_results = pd.DataFrame(results).sort_values(by="AIC")
        st.dataframe(df_results)



                    



