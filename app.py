# File: app.py

import streamlit as st
st.title("✅ App đang chạy ngon lành!")

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_main.csv")
    df['explicit'] = df['explicit'].astype(int)

    label_cols = ['track_id', 'artists', 'album_name', 'track_name']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df['track_genre'] = LabelEncoder().fit_transform(df['track_genre'].astype(str))

    numeric_cols = [
        'popularity', 'duration_ms', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]
    X = df[label_cols + ['explicit'] + numeric_cols]
    y = df['track_genre']
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

X_scaled, y = load_data()

# Sidebar control
task = st.sidebar.selectbox("Chọn tác vụ", ["Classification", "Clustering"])

if task == "Classification":
    model_name = st.sidebar.selectbox("Chọn mô hình", ["k-NN (auto)", "k-NN (Ball Tree)", "Random Forest"])
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    start = time.time()
    if model_name == "k-NN (auto)":
        model = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    elif model_name == "k-NN (Ball Tree)":
        model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    duration = time.time() - start
    acc = accuracy_score(y_test, y_pred)

    st.write(f"### 🔍 Kết quả mô hình: {model_name}")
    st.write(f"- Accuracy: `{acc:.4f}`")
    st.write(f"- Thời gian dự đoán: `{duration:.4f} s`")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_pred, palette="Set1", legend=False, ax=ax)
    plt.title(model_name)
    st.pyplot(fig)

elif task == "Clustering":
    k = st.sidebar.slider("Số cụm (k)", 2, 10, 5)
    algo = st.sidebar.selectbox("Thuật toán", ["KMeans"])

    start = time.time()
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    duration = time.time() - start
    silhouette = silhouette_score(X_scaled, labels)

    st.write(f"### 📊 Kết quả phân cụm")
    st.write(f"- Silhouette Score: `{silhouette:.4f}`")
    st.write(f"- Thời gian phân cụm: `{duration:.4f} s`")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="Set2", legend=False, ax=ax)
    plt.title("Clustering Visualization")
    st.pyplot(fig)
