# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:50:53 2025

@author: wt928
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("名前一覧.xlsx") #男性:2554件(約59％),女性:1772件(約41％)
df = df.drop(df.columns[2] , axis=1)

vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
X = vectorizer.fit_transform(df["名前"])
Y = df["性別"].map({"男性": 0, "女性": 1})  # 男性=0, 女性=1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #訓練データ80％、テストデータ20％
model = LogisticRegression()
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print(f"精度: {accuracy:.2%}")

def predict_gender(name):
    name_vec = vectorizer.transform([name])
    prob = model.predict_proba(name_vec)[0]  # [男性確率, 女性確率]
    return {"Male": prob[0], "Female": prob[1]}

name = input("名前を入力してください:")
result = predict_gender(name)
print(f"男性: {result['Male']:.2%}, 女性: {result['Female']:.2%}")



