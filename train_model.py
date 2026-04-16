import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD DATA (DI-RINGANKAN)
# =========================
df = pd.read_csv("airlines_flights_data.csv")

# 🔥 SAMPLE DATA BIAR TIDAK MACET
df = df.sample(n=min(5000, len(df)), random_state=42)

target = "price" if "price" in df.columns else df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

# =========================
# PREPROCESS
# =========================
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# =========================
# MODEL (RINGAN & CEPAT)
# =========================
model = RandomForestRegressor(
    n_estimators=40,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# TRAIN
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, pred)

# =========================
# SAVE MODEL
# =========================
joblib.dump(pipeline, "model_tiket.pkl")
joblib.dump({
    "model": "RandomForest_Fast",
    "mae": mae
}, "model_meta.pkl")

print(f"✅ Training selesai | MAE: {mae}")