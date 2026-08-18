import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv("student_performance.csv")
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

X = df.drop(columns="passed")
y = df["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, random_state=42)
}

results, roc_info = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append([name, accuracy_score(y_test,pred), precision_score(y_test,pred,zero_division=0),
                    recall_score(y_test,pred,zero_division=0), f1_score(y_test,pred,zero_division=0),
                    roc_auc_score(y_test,proba)])
    roc_info[name] = roc_curve(y_test, proba)

results = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score","ROC AUC"]).sort_values("ROC AUC",ascending=False)
print("\nMODEL PERFORMANCE\n", results.round(3).to_string(index=False))
results.to_csv("model_results.csv",index=False)

best_name = results.iloc[0]["Model"]
cm = confusion_matrix(y_test, models[best_name].predict(X_test))
print("\nBest model:", best_name)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,5)); plt.imshow(cm); plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.xticks([0,1],["Fail","Pass"]); plt.yticks([0,1],["Fail","Pass"])
for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
plt.tight_layout(); plt.savefig("confusion_matrix.png",dpi=180); plt.show()

plt.figure(figsize=(7,5))
for name,(fpr,tpr,_) in roc_info.items():
    auc=results.loc[results.Model==name,"ROC AUC"].iloc[0]
    plt.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout(); plt.savefig("roc_curves.png",dpi=180); plt.show()

plt.figure(figsize=(8,5)); plt.bar(results["Model"],results["Accuracy"]); plt.ylim(0,1)
plt.ylabel("Accuracy"); plt.title("Model Accuracy Comparison"); plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig("model_comparison.png",dpi=180); plt.show()
