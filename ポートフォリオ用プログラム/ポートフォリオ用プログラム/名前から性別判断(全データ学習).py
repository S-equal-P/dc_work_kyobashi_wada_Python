# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("名前一覧.xlsx") #男性:2554件(約59％),女性:1773件(約41％)
df = df.drop(df.columns[2] , axis=1) #出典URLの削除

vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3)) #n-gram:1文字～3文字
X = vectorizer.fit_transform(df["名前"])
Y = df["性別"].map({"男性": 0, "女性": 1})  # 男性=0, 女性=1

model = LogisticRegression() #ロジスティック回帰
model.fit(X, Y)

def predict_gender(name):
    name_vec = vectorizer.transform([name])
    prob = model.predict_proba(name_vec)[0]
    return {"Male": prob[0], "Female": prob[1]} #[男性確率, 女性確率]

while True:
   name = input("ひらがなで名前を入力してください。\n※止める場合は「やめる」と入力して下さい:")
   if name == "やめる":
       break
   result = predict_gender(name)
   print(f"男性: {result['Male']:.2%}, 女性: {result['Female']:.2%}" + "\n")