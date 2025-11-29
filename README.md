# 自動データ分析アプリ「Auto-Fitter for Lab」
アプリへのリンクはこちら</br>
https://auto-fitter-mpo8fnkwdmgvnsegxxu8aw.streamlit.app/  </br>

アプリ使用時の映像</br>

https://github.com/user-attachments/assets/9cc3eac4-9732-4844-9fcb-494fca1ec32b


## 理系学生のための実験データ解析・考察支援ツール
大学の実験レポート作成において、最も手間のかかる回帰分析と結果の科学的解釈を自動化するために開発しました。
## アプリ開発に至った背景
私の通う大学では必修に実験科目があるのですが、不正防止の観点からとにかくアナログでうんざりしていた。（関数電卓のよくわからないコマンドを使って回帰直線求めさせるなど）
こうした課題を**pythonの数値計算ライブラリ**と**LLM**を組み合わせることで解決しました。
## 主な機能
- 回帰直線または回帰曲線の導出 </br>
基本的な線形、二次関数、指数関数、対数関数。さらに理系のドメインを意識したシグモイド関数、ミカエリス・メンテン式にも対応。
- 統計的モデル選択</br>
AICcを用いて過学習を防ぎつつ最適なモデルを自動ランク付け (AICだと実験データが少ない場合に過学習が起こりやすくなってしまったので、パラメータ増加によるペタルティーがおもいAICcに変更しました。)
- 不確かさの可視化</br>
モンテカルロ法による95％信頼区間の描画</br>
パラメータの標準誤差の算出
- AIによる考察生成</br>
解析結果とユーザーが入力したデータの背景情報を組み合わせ、gemini APIを呼び出し考察案を生成。
## 使用技術
- Laguage: python
- Framework: stremalit
- DataScience: scipy.optimize(非線形最小二乗法）,numpy, pandas, matplotlib
- AI: google gemini API
## ローカルでの実行方法
1. リポジトリのクローン</br>
  git clone https://github.com/guten-morgen3776/auto-fitter.git</br> 
2. 依存ライブラリのインストール</br>
  pip install -r requirement.txt </br>
3. APIキーの設定</br>
  google AI studioからAPIキーを取得し、プロジェクト直下に.streamlit/secrets.tomlファイルを作成して、以下のように記述してください。
```
GREMINI_API_KEY = 'ここにあなたのAPIキーを貼り付けてください'
```
4. アプリの実行
streamlit -m run app.py

