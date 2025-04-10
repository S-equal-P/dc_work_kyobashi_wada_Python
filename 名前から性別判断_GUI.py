# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui import Ui_MainWindow  # pyuic5で変換したGUIクラス
import joblib
import re

# PyInstaller対応のファイルパス取得
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

model_path = os.path.join(base_path, "gender_model.pkl")
vectorizer_path = os.path.join(base_path, "vectorizer.pkl")

# モデルとベクトライザーの読み込み
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#入力された文字列から確率を出力
def predict_gender(name):
    name_vec = vectorizer.transform([name])
    prob = model.predict_proba(name_vec)[0]
    return {"Male": prob[0], "Female": prob[1]}

class GenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(450, 350)
        # ボタンイベント設定
        self.ui.pushButton.clicked.connect(self.on_predict)

    def on_predict(self):
        name = self.ui.lineEdit.text()

        # 入力バリデーション（ひらがなかチェック）
        hiragana = re.compile('[\u3041-\u309F]+')
        if hiragana.fullmatch(name) is None:
            self.ui.textEdit.setPlainText("※ひらがな以外の文字列が入力されました！")
            return

        result = predict_gender(name)
        output = f"男性: {result['Male']:.2%}\n女性: {result['Female']:.2%}"
        self.ui.textEdit.setPlainText(output)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenderApp()
    window.show()
    sys.exit(app.exec_())
