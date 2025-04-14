# 名前から性別判断アプリ
## 概要
このアプリケーションは、入力された日本の名前から性別を判定するツールです。機械学習モデルを使用しており、ユーザーは直感的なGUIを通じて操作できます。​

## 機能
・名前の入力による性別判定​

・GUIを用いた直感的な操作​

## 使用方法
必要なライブラリのインストール：

以下のコマンドを実行して、必要なライブラリをインストールします。

```bash
 pip install -r requirements.txt
```

アプリケーションの起動：

以下のコマンドでアプリケーションを起動します。

```bash
python 名前から性別判断_GUI.py
```

GUIの操作方法：

![MainWindow 2025_04_14 10_53_40](https://github.com/user-attachments/assets/91e99106-c0f2-4939-8df9-67d74837db8e)

・分析開始ボタンの横にあるテキストボックスに分析したい名前を入力する

・分析開始ボタンをクリックする

・下のテキストボックスに「男性：○○ ％、女性：○○ ％」という形で、その名前がどちらの性別の可能性が高いかを出力します

※ひらがな以外の文字列を入力した場合、エラー文が出力されます

## ファイル構成

・名前から性別判断_GUI.py：​メインのGUIアプリケーション​

・gui.ui、gui.py：GUIフォーマット(gui.pyは名前から性別判断_GUI.py実行時に必要)

・名前から性別判断_学習.py：名前一覧.xlsxを学習データとしてgender_model.pkl、vectorizer.pklを生成

・名前一覧.xlsx：ひらがなの名前と性別の対応表

・gender_model.pkl：​性別判定モデル​

・vectorizer.pkl：​特徴量変換器​

・requirements.txt：​必要なライブラリ一覧​

## ライセンス
このプロジェクトはMITライセンスのもとで公開されています。​




