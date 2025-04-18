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

### 必要なライブラリのインストール

以下のコマンドを実行して、必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### アプリケーションの起動

以下のコマンドでアプリケーションを起動します。

```bash
python 名前から性別判断_GUI.py
```

---

## GUIの操作方法

![MainWindow 2025_04_14 10_53_40](https://github.com/user-attachments/assets/91e99106-c0f2-4939-8df9-67d74837db8e)

- 「分析開始」ボタンの横にあるテキストボックスに、分析したい名前（ひらがな）を入力
- 「分析開始」ボタンをクリック
- 下のテキストボックスに「男性：○○％、女性：○○％」の形式で判定結果が表示されます

※ ひらがな以外の文字列を入力した場合、エラー文が表示されます

---

## 性別を予測する仕組み

このアプリでは、名前（ひらがな）から性別を確率的に推定するモデルを使用しています。  
アルゴリズムには、`scikit-learn` の **ロジスティック回帰（Logistic Regression）** を採用しています。

### 処理の流れ

1. **n-gram分割**  
　名前を1～3文字単位のn-gramに分割  
　例：「さくら」→「さ」「く」「ら」「さく」「くら」「さくら」

2. **ベクトル化**  
　`CountVectorizer` を使い、n-gramの出現パターンを数値ベクトルに変換

3. **モデル学習**  
　`名前から性別判断_学習.py` にて、`名前一覧.xlsx` を用いてロジスティック回帰モデルを学習


学習に使用した `名前一覧.xlsx` は、以下のような形式で構成されています：

| 名前   | 性別 |
|--------|------|
| たろう | 男   |
| ひかる | 男   |
| さくら | 女   |
| ひかる | 女   |

- 同じ名前でも複数の性別ラベルが存在する場合があります（例：「ひかる」など）
- 「男」「女」のラベルに基づいて、名前に含まれる文字列パターンと性別の関係をモデルが学習します

4. **確率予測**  
　`名前から性別判断_GUI.py` にて、学習済みモデルを読み込み  
　`predict_proba` を使って「男性である確率」「女性である確率」を出力

　   例：「ひかる」のように中性的な名前では、`男性 48% / 女性 52%` のように近い値になることがあります。

    本アプリケーションで表示される「男性：○○％ / 女性：○○％」は、  
    ロジスティック回帰モデルが学習データをもとに推定した**確率的な分類結果**です。
    
    これは、「この名前が学習データにおいてどの性別と関連していたか」に基づいて推定されたもので、  
    必ずしも現実の性別と一致するとは限りません。  
    特に中性的な名前では、確率が50%前後に近づくことがあります。
    
    出力例：
    - 「たろう」 → 男性 96% / 女性 4%
    - 「さくら」 → 男性 5% / 女性 95%
    - 「ひかる」 → 男性 48% / 女性 52%
    

### 備考

- 入力はひらがなのみを対象とし、漢字・カタカナ・英字には対応していません
- モデルの再学習は `名前から性別判断_学習.py` でのみ行われます
- 学習済みモデルは `.pkl` ファイルとして保存され、通常の使用ではそれを読み込みます

---

## 📊 確率の意味と出力の解釈



---

## ファイル構成

- `名前から性別判断_GUI.py`：メインのGUIアプリケーション
- `gui.ui`, `gui.py`：GUIレイアウトファイル（`pyuic5` 変換済）
- `名前から性別判断_学習.py`：学習スクリプト（モデルとベクトライザ生成）
- `名前一覧.xlsx`：ひらがなの名前と性別ラベルの対応表
- `gender_model.pkl`：学習済みロジスティック回帰モデル
- `vectorizer.pkl`：CountVectorizerによる文字ベクトル変換器
- `requirements.txt`：必要なライブラリ一覧

---

## ライセンス

このプロジェクトは MITライセンス のもとで公開されています。



