import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# データ読み込み
df = pd.read_csv("race_data.csv")

# 特徴量と正解ラベルを分ける
X = df[["distance", "weight", "jockey_rank", "popularity"]]
y = df["win"]

# 訓練用とテスト用に分割（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストで学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度を表示
print("正解率：", accuracy_score(y_test, y_pred))


# 新しいレース（仮）のデータ
new_horse = pd.DataFrame({
    "distance": [1600],
    "weight": [56],
    "jockey_rank": [2],
    "popularity": [1]
})

# 勝つ確率を予測
win_proba = model.predict_proba(new_horse)[0][1]
print(f"この馬が勝つ確率は：{win_proba:.2f}")