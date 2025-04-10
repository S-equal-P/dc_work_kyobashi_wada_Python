import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#モデルとベクトライザーの保存先
MODEL_PATH = "gender_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# モデルとベクトライザーの学習
try:
    df = pd.read_excel("名前一覧.xlsx") #男性:2554件(約59％),女性:1773件(約41％)
    df = df.drop(df.columns[2] , axis=1) #出典URLの削除
except FileNotFoundError:
    print("エラー: '名前一覧.xlsx' が見つかりません。")
    exit(1)

vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3)) #n-gram:1文字～3文字
X = vectorizer.fit_transform(df["名前"])
Y = df["性別"].map({"男性": 0, "女性": 1}) #男性=0, 女性=1

model = LogisticRegression() #ロジスティック回帰

#5分割交差検証の実行
scores = cross_val_score(model, X, Y, cv=5, scoring="accuracy") 
print(f"各フォールドの正解率: {[f'{score:.2%}' for score in scores]}")
print(f"平均正解率: {scores.mean():.2%}")
print(f"標準偏差: {scores.std():.4f}")

model.fit(X, Y) #全データ学習
    
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print("モデルとベクトライザーを保存しました。")
    




