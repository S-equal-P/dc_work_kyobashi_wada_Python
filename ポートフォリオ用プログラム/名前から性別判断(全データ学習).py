# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:19:02 2025

@author: wt928
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("名前一覧.xlsx") #男性:2554件(約59％),女性:1772件(約41％)
df = df.drop(df.columns[2] , axis=1)

vectorizer = CountVectorizer(analyzer="char")
X = vectorizer.fit_transform(df["名前"])
Y = df["性別"].map({"男性": 0, "女性": 1})  # 男性=0, 女性=1

model = LogisticRegression()
model.fit(X, Y)

def predict_gender(name):
    name_vec = vectorizer.transform([name])
    prob = model.predict_proba(name_vec)[0]  # [男性確率, 女性確率]
    return {"Male": prob[0], "Female": prob[1]}

while True:
   name = input("名前を入力してください(ひらがな):")
   result = predict_gender(name)
   print(f"男性: {result['Male']:.2%}, 女性: {result['Female']:.2%}")
   print("")