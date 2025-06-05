import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# App Title
st.title("Real-Time Credit Card Fraud Detection")
st.write("A web app to detect fraudulent transactions using a Decision Tree Classifier.")

# Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file containing transaction data.", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(data.head())
    
    # Preprocess the data
    st.write("### Preprocessing the Data:")
    if 'Time' in data.columns:
        data = data.drop(columns=['Time'])
        st.write("Dropped 'Time' column.")
    
    if 'Amount' in data.columns:
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data[['Amount']])
        st.write("Scaled 'Amount' column.")
    
    if data.duplicated().any():
        initial_shape = data.shape
        data = data.drop_duplicates()
        st.write(f"Dropped duplicates. Reduced dataset size from {initial_shape} to {data.shape}.")
    
    if data.isnull().sum().sum() > 0:
        st.write("The dataset contains missing values. Please clean the dataset before uploading.")
        st.stop()

    # Split the data into features and labels
    X = data.drop('Class', axis=1)
    Y = data['Class']
    
    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train the model
    st.write("Training a Decision Tree Classifier...")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    
    # Evaluate the model
    Y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    
    st.write("### Model Performance:")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

    # Allow user input for prediction
    st.write("### Predict Fraud for New Transactions:")
    input_values = st.text_input(
        f"Enter all feature values separated by commas (e.g., -0.425965,0.960523,...):"
    )

    if st.button("Predict"):
        try:
            # Convert input values to a numpy array
            input_array = np.array([float(x) for x in input_values.split(",")]).reshape(1, -1)
            
            if input_array.shape[1] != X.shape[1]:
                st.error(f"Expected {X.shape[1]} features but got {input_array.shape[1]}. Please check your input.")
            else:
                # Make prediction
                prediction = dtc.predict(input_array)
                if prediction[0] == 0:
                    st.success("The transaction is Normal.")
                else:
                    st.error("The transaction is Fraudulent.")
        except ValueError:
            st.error("Invalid input. Please enter numerical values separated by commas.")
else:
    st.write("Please upload a dataset to proceed.")
