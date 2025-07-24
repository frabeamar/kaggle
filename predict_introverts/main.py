import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    df = pd.read_csv("data/introverts/train.csv")
    # test = pd.read_csv("data/introverts/test.csv")
    labels = LabelEncoder().fit_transform(df["Personality"])

    df = df.drop("Personality", axis=1)

    cat_cols = df.select_dtypes(include=["object"]).columns

    encode = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                df.select_dtypes(include=["float64", "int64"]).columns,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("encode", encode),
            ("impute", IterativeImputer()),
        ]
    )

    X = pipeline.fit_transform(df)

    x_train, x_test, y_train, y_test = train_test_split(X, labels)
    reg = GaussianProcessClassifier()
    reg.fit(x_train, y_train)
    pred_train = reg.predict(x_train)
    pred = reg.predict(x_test)

    acc = accuracy_score(y_test, pred)
