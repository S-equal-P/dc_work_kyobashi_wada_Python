# 名前から性別判断アプリ

## 概要

このアプリケーションは、入力された日本の名前（ひらがな）から性別を確率的に判定するGUIツールです。  
Pythonと機械学習（ロジスティック回帰）を用いて構築されており、ユーザーは直感的な操作で判定結果を得ることができます。

---

## 主な機能

- 名前の入力による性別判定（男性・女性の確率を出力）
- GUIによるシンプルで直感的な操作
- 入力バリデーション（ひらがな以外には警告表示）

---

## 使用方法

### ライブラリのインストール

```bash
pip install -r requirements.txt
```

### アプリケーションの起動

```bash
python 名前から性別判断_GUI.py
```

---

## GUIの操作方法

![MainWindow 2025_04_14 10_53_40](https://github.com/user-attachments/assets/91e99106-c0f2-4939-8df9-67d74837db8e)

1. テキストボックスに分析したい名前（ひらがな）を入力  
2. 「分析開始」ボタンをクリック  
3. 判定結果が下部に「男性：○○％、女性：○○％」の形式で表示されます

※ ひらがな以外を入力した場合は警告が表示されます。

---

## 性別を予測する仕組み

このアプリでは、名前（ひらがな）から性別を確率的に推定するモデルを使用しています。  
アルゴリズムには、`scikit-learn` の **ロジスティック回帰（Logistic Regression）** を採用しています。

### 処理の流れ

1. **n-gram分割**  
　名前を1～3文字単位のn-gramに分割  
　例：「さくら」→「さ」「く」「ら」「さく」「くら」「さくら」

2. **ベクトル化**  
　`CountVectorizer` によってn-gramの出現パターンを数値ベクトルに変換

3. **モデル学習**  
　`名前から性別判断_学習.py` にて、`名前一覧.xlsx` を用いてロジスティック回帰モデルを学習します。

　使用データの例：

　| 名前   | 性別 |
　|--------|------|
　| たろう | 男   |
　| ひかる | 男   |
　| さくら | 女   |
　| ひかる | 女   |

　※ 同じ名前に複数の性別が対応することがあります（中性的な名前など）

4. **確率予測**  
　`名前から性別判断_GUI.py` にて、学習済みモデルを用いて判定を行います。  
　`predict_proba()` により、以下のように性別ごとの確率が出力されます。

　例：  
　・「たろう」 → 男性 96% / 女性 4%  
　・「さくら」 → 男性 5% / 女性 95%  
　・「ひかる」 → 男性 48% / 女性 52%

　この確率は、学習データに基づいてロジスティック回帰モデルが推定したもので、  
　「この名前が過去のデータにおいてどの性別と関連が強かったか」を表す統計的な値です。  
　現実の性別と必ず一致するものではありません。

---

## ファイル構成

- `名前から性別判断_GUI.py`：GUI本体
- `gui.ui`, `gui.py`：Qt DesignerベースのGUIレイアウト
- `名前から性別判断_学習.py`：モデル学習スクリプト
- `名前一覧.xlsx`：ひらがな名前と性別ラベルの表
- `gender_model.pkl`：学習済みロジスティック回帰モデル
- `vectorizer.pkl`：CountVectorizerによるベクトル変換器
- `requirements.txt`：必要ライブラリ一覧

---

## ライセンス

このプロジェクトは MITライセンス のもとで公開されています。



