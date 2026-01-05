import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("atlantis_citizens_final.csv")

data = data.drop(columns=["Citizen_ID", "Bio_Hash"])

data["Wealth_Index"] = data.groupby("District_Name")["Wealth_Index"] \
                            .transform(lambda x: x.fillna(x.median()))

data["House_Size_sq_ft"] = data.groupby("District_Name")["House_Size_sq_ft"] \
                               .transform(lambda x: x.fillna(x.median()))

data["Life_Expectancy"] = data.groupby("District_Name")["Life_Expectancy"] \
                               .transform(lambda x: x.fillna(x.median()))

output_encoding = {
    "Warrior": 0,
    "Merchant": 1,
    "Fisher": 2,
    "Miner": 3,
    "Scribe": 4
}

X = data.drop(columns=["Occupation"])
y = data["Occupation"].map(output_encoding)
numerical_cols = ["Wealth_Index", "House_Size_sq_ft", "Life_Expectancy"]
categorical_col = ["Diet_Type", "District_Name", "Vehicle_Owned", "Work_District"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_col)
    ]
)

model = LogisticRegression(
    max_iter=4000,
    solver="lbfgs",
    class_weight="balanced",
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

pipeline.fit(X, y)
test_data = pd.read_csv("test_atlantis_hidden.csv")

X_test = test_data.drop(columns=["Citizen_ID", "Bio_Hash"])
y_test_pred = pipeline.predict(X_test)
submission = pd.DataFrame({
    "Citizen_ID": test_data["Citizen_ID"],
    "Occupation": y_test_pred
})

submission.to_csv("vedu_prediction.csv", index=False)