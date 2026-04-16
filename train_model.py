import pandas as pd  # type: ignore
import joblib   # type: ignore

from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

try:
    from xgboost import XGBRegressor    # type: ignore
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

df = pd.read_csv("airlines_flights_data.csv")

target = "price" if "price" in df.columns else df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

cat_cols = X.select_dtypes(include=["object", "string"]).columns
num_cols = X.select_dtypes(exclude=["object", "string"]).columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

def build(model_type):
    if model_type == "xgb" and XGB_AVAILABLE:
        model = XGBRegressor(n_estimators=200, max_depth=6)
    else:
        model = RandomForestRegressor(n_estimators=120)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {}

rf = build("rf")
rf.fit(X_train, y_train)
models["RandomForest"] = (rf, mean_absolute_error(y_test, rf.predict(X_test)))

if XGB_AVAILABLE:
    xgb = build("xgb")
    xgb.fit(X_train, y_train)
    models["XGBoost"] = (xgb, mean_absolute_error(y_test, xgb.predict(X_test)))

best = min(models, key=lambda k: models[k][1])
model, mae = models[best]

joblib.dump(model, "model_tiket.pkl")
joblib.dump({"model": best, "mae": mae}, "model_meta.pkl")

print(f"Best Model: {best} | MAE: {mae}")