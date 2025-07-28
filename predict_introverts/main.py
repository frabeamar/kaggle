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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Linear SVM": SVC(C=0.025, kernel="linear", random_state=42),
        "RBF SVM": SVC(C=1, gamma=2, random_state=42),
        "Gaussian Process": GaussianProcessClassifier(
            kernel=1**2 * RBF(length_scale=1), random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            max_depth=5, max_features=1, n_estimators=10, random_state=42
        ),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }
    

    x_train, x_test, y_train, y_test = train_test_split(X, labels)
    result = []
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        acc = accuracy_score(y_test, pred)
        result.append({"classifier": name, "accuracy": acc})

