import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# =========================
# Load and Clean Dataset
# =========================
df = pd.read_csv("MERGED.csv")
df.columns = df.columns.str.strip()
df["Recommended Crop"] = df["Recommended Crop"].astype(str).str.strip().str.title()

# =========================
# Dataset Size Summary
# =========================
print(f"Number of observations (rows): {df.shape[0]}")
print(f"Number of features (columns): {df.shape[1]}")

# =========================
# Exploratory Data Analysis
# =========================

# Countplot with value labels
plt.figure(figsize=(16, 6))
ax = sns.countplot(
    data=df,
    x="Recommended Crop",
    order=df["Recommended Crop"].value_counts().index,
    palette="dark",
)
plt.title("Number of Records per Recommended Crop")
plt.xticks(rotation=45)

# Add value labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2.0, height),
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()

print("\nDataset Overview:")
print(df.head())
print(df.tail())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# =========================
# Histograms with Axis Labels + Mean/Median Lines
# =========================
axes = df.hist(figsize=(14, 10), bins=20, edgecolor="black")
plt.suptitle("Histograms of Numerical Features", fontsize=16)

# Loop through each subplot and add labels + reference lines
for ax in axes.flatten():
    feature = ax.get_title()           # pandas puts the column name in the title
    if feature == "":                  # skip empty axes (if any)
        continue
    data = df[feature].dropna().values

    # Axis labels
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("")  # remove duplicate title (since we now use x-label)

    # Mean/Median reference lines
    mean_val = np.mean(data)
    median_val = np.median(data)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean={mean_val:.2f}")
    ax.axvline(median_val, color="blue", linestyle=":", linewidth=1, label=f"Median={median_val:.2f}")
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # keep space for suptitle
plt.show()


# Boxplots for each numerical feature grouped by crop
for col in ["N", "P", "K", "temperature", "humidity", "ph", "Annual Rainfall"]:
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df, x="Recommended Crop", y=col)
    plt.title(f"{col} by Crop")
    plt.xticks(rotation=45)

    # Calculate and annotate outliers as numeric labels only
    outlier_counts = df.groupby("Recommended Crop")[col].apply(
        lambda x: (
            (x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
            | (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
        ).sum()
    )

    for i, crop in enumerate(df["Recommended Crop"].unique()):
        if crop in outlier_counts:
            count = outlier_counts[crop]
            ax.annotate(
                f"{count}",
                xy=(i, df[col].quantile(0.95)),
                ha="center",
                fontsize=9,
                color="darkred",
                xytext=(0, 8),
                textcoords="offset points",
            )

    # Legend-like note below the x-axis
    plt.figtext(
        0.5,
        -0.1,
        "Numbers above boxes represent outlier counts",
        ha="center",
        fontsize=10,
        color="darkred",
    )
    plt.tight_layout()
    plt.show()

# =========================
# Interactive Scatter Plots
# =========================
fig3 = px.scatter(
    df,
    x="humidity",
    y="Annual Rainfall",
    color="Recommended Crop",
    title="Scatter: Annual Rainfall vs Humidity by Crop",
    labels={"humidity": "Humidity (%)", "Annual Rainfall": "Annual Rainfall (mm)"},
    template="plotly",
    height=600,
)
fig3.show()
fig3.write_html("rainfall_vs_humidity_scatter.html")

fig1 = px.scatter(
    df,
    x="temperature",
    y="Annual Rainfall",
    color="Recommended Crop",
    title="Scatter: Annual Rainfall vs Temperature by Crop",
    labels={"temperature": "Temperature (°C)", "Annual Rainfall": "Annual Rainfall (mm)"},
    template="plotly",
    height=600,
)
fig1.show()
fig1.write_html("rainfall_vs_temperature_scatter.html")

fig2 = px.scatter(
    df,
    x="humidity",
    y="Annual Rainfall",
    color="Recommended Crop",
    title="Scatter: Annual Rainfall vs Humidity by Crop",
    labels={"humidity": "Humidity (%)", "Annual Rainfall": "Annual Rainfall (mm)"},
    template="plotly",
    height=600,
)
fig2.show()
fig2.write_html("rainfall_vs_humidity_scatter.html")

# =========================
# Encode Target
# =========================
label_encoder = LabelEncoder()
df["Encoded Crop"] = label_encoder.fit_transform(df["Recommended Crop"])

# =========================
# Feature and Target Split
# =========================
features = ["N", "P", "K", "temperature", "humidity", "ph", "Annual Rainfall"]
target = "Encoded Crop"

X = df[features]
y = df[target]

# =========================
# Stratified Train-Test Split
# =========================
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in split.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# =========================
# Scale Features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Prepare for Deep Learning Models
# =========================
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# =========================
# Train Multiple Models
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
}

model_performance = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    model_performance[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# =========================
# Train CNN
# =========================
cnn = Sequential(
    [
        Input(shape=(X_train_cnn.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation="relu"),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

cnn.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(X_train_cnn, y_train_cat, epochs=50, batch_size=16, verbose=0)
cnn_loss, cnn_acc = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
model_performance["CNN"] = cnn_acc
print(f"CNN Accuracy: {cnn_acc:.4f}")

# =========================
# Visualize Model Performance
# =========================
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

metrics_data = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    metrics_data.append(
        {"Model": name, "Accuracy": model_performance[name], "Precision": precision, "Recall": recall, "F1 Score": f1}
    )

# CNN metrics
y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)
precision_cnn = precision_score(y_test, y_pred_cnn, average="macro", zero_division=0)
recall_cnn = recall_score(y_test, y_pred_cnn, average="macro", zero_division=0)
f1_cnn = f1_score(y_test, y_pred_cnn, average="macro", zero_division=0)
metrics_data.append(
    {"Model": "CNN", "Accuracy": model_performance["CNN"], "Precision": precision_cnn, "Recall": recall_cnn, "F1 Score": f1_cnn}
)

metrics_df = pd.DataFrame(metrics_data)

# Barplot of all metrics
metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(12, 7), colormap="Set2"
)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_performance_metrics.png")
plt.show()

# =========================
plt.figure(figsize=(10, 6))
sns.barplot(
    x=list(model_performance.keys()),
    y=list(model_performance.values()),
    hue=list(model_performance.keys()),
    palette="viridis",
    legend=False,
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# Confusion Matrices
# =========================
from sklearn.metrics import ConfusionMatrixDisplay

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=label_encoder.classes_, xticks_rotation=45, cmap="Blues"
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

# CNN confusion matrix
y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_cnn, display_labels=label_encoder.classes_, xticks_rotation=45, cmap="Blues"
)
plt.title("Confusion Matrix - CNN")
plt.tight_layout()
plt.show()

# =========================
# Select Best Model
# =========================
best_model_name = max(model_performance, key=model_performance.get)
print(f"\nBest Model: {best_model_name} with Accuracy: {model_performance[best_model_name]:.4f}")

# =========================
# Save Best Model, Scaler, and Encoder
# =========================
if best_model_name == "CNN":
    cnn.save("best_crop_model_cnn.h5")
else:
    with open("best_crop_model.pkl", "wb") as f:
        pickle.dump(models[best_model_name], f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Best model, scaler, and encoder saved.")

# ============================
# === RL INTEGRATION START ===
# ============================
def _inject_rl_layer():
    import pickle
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import Dict
    from rl_agent import ContextualBanditAgent

    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        tf = None

    DATA_PATH = Path("MERGED.csv")
    SCALER_PATH = Path("scaler.pkl")
    ENCODER_PATH = Path("label_encoder.pkl")
    SK_MODEL_PATH = Path("best_crop_model.pkl")
    CNN_MODEL_PATH = Path("best_crop_model_cnn.h5")
    AGENT_PATH = Path("rl_agent.pkl")

    assert DATA_PATH.exists(), "MERGED.csv not found."
    assert SCALER_PATH.exists() and ENCODER_PATH.exists(), "Missing scaler.pkl or label_encoder.pkl."

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    if SK_MODEL_PATH.exists():
        with open(SK_MODEL_PATH, "rb") as f:
            best_model = pickle.load(f)
        use_cnn = False
    else:
        assert CNN_MODEL_PATH.exists(), "No best model file found."
        assert tf is not None, "TensorFlow not available to load CNN model."
        best_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        use_cnn = True

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["Recommended Crop"] = df["Recommended Crop"].astype(str).str.strip().str.title()
    features = ["N", "P", "K", "temperature", "humidity", "ph", "Annual Rainfall"]
    X = df[features].values
    y = label_encoder.transform(df["Recommended Crop"].values)
    X_scaled = scaler.transform(X)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=123, stratify=y)

    def base_policy_proba(model, Xb):
        if use_cnn:
            Xb_cnn = Xb.reshape(-1, Xb.shape[1], 1)
            return model.predict(Xb_cnn, verbose=0)
        else:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(Xb)
            else:
                return np.ones((Xb.shape[0], len(label_encoder.classes_))) / len(label_encoder.classes_)

    agent = ContextualBanditAgent(
        n_actions=len(label_encoder.classes_),
        feature_dim=X_tr.shape[1],
        epsilon=0.10,
        random_state=123,
    )

    proba_val = base_policy_proba(best_model, X_val)
    actions, _ = agent.choose_action(X_val, base_policy_proba=proba_val)
    rewards = (actions == y_val).astype(float)
    agent.update(X_val, actions, rewards)
    agent.save(str(AGENT_PATH))

    def _features_to_array(features_dict: Dict[str, float]):
        order = ["N", "P", "K", "temperature", "humidity", "ph", "Annual Rainfall"]
        return scaler.transform(np.array([features_dict[k] for k in order], dtype=np.float64)[None, :])

    def recommend(features_dict: Dict[str, float]):
        x = _features_to_array(features_dict)
        p = base_policy_proba(best_model, x)
        action, _ = agent.choose_action(x, base_policy_proba=p)
        crop_name = label_encoder.inverse_transform([action[0]])[0]
        return crop_name, int(action[0])

    def update_with_feedback(features_dict: Dict[str, float], chosen_crop: str, reward: float):
        x = _features_to_array(features_dict)
        action_id = int(label_encoder.transform([chosen_crop])[0])
        agent.update(x, np.array([action_id]), np.array([float(reward)]))
        agent.save(str(AGENT_PATH))

    print("RL warm-start complete. Agent saved to rl_agent.pkl")
    print("Use recommend(features_dict) and update_with_feedback(features_dict, chosen_crop, reward)")
    return {"recommend": recommend, "update_with_feedback": update_with_feedback}

if __name__ == "__main__":
    try:
        _ = _inject_rl_layer()
    except Exception as e:
        print("[RL integration skipped]", e)
# ==========================
# === RL INTEGRATION END ===
# ==========================
