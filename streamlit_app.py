import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data(filename):
    return pd.read_csv('data/passing_cleaned.csv')

# Normalization functions
def z_score_normalization(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def min_max_normalization(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Classification models
def train_model(model_name, X_train, y_train, **kwargs):
    if model_name == 'Random Forest':
        model = RandomForestClassifier(**kwargs)
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier(**kwargs)
    elif model_name == 'SVM':
        model = SVC(**kwargs)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(**kwargs)
    
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    st.title("Machine Learning Playground")

    # Sidebar - File Upload
    st.sidebar.title("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write(data)

        # Sidebar - Normalization
        st.sidebar.title("Normalization")
        normalization_method = st.sidebar.selectbox("Choose normalization method", ("Z-score", "Min-Max"))

        if normalization_method == "Z-score":
            normalized_data = z_score_normalization(data)
        elif normalization_method == "Min-Max":
            normalized_data = min_max_normalization(data)

        # Sidebar - Model Selection
        st.sidebar.title("Classification Models")
        model_name = st.sidebar.selectbox("Choose a model", ("Random Forest", "AdaBoost", "SVM", "Decision Tree"))

        # Sidebar - Model Hyperparameters
        st.sidebar.title("Model Hyperparameters")
        if model_name == "Random Forest":
            n_estimators = st.sidebar.slider("Number of estimators", 1, 100, 10)
            max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
            model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        elif model_name == "AdaBoost":
            n_estimators = st.sidebar.slider("Number of estimators", 1, 100, 50)
            learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.1)
            model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        elif model_name == "SVM":
            C = st.sidebar.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"))
            model_params = {'C': C, 'kernel': kernel}
        elif model_name == "Decision Tree":
            max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
            model_params = {'max_depth': max_depth}

        # Train-test split
        X = normalized_data.drop(columns=["target_column"])
        y = normalized_data["target_column"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate the model
        model = train_model(model_name, X_train, y_train, **model_params)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write("Accuracy:", accuracy)


if __name__ == "__main__":
    main()






# Page title
#st.set_page_config(page_title='Quarterback Statistics 2001-2023')
#st.title('Quarterback Statistics 2001-2023')

# Load data
#df = pd.read_csv('data/passing_cleaned.csv')
#df
