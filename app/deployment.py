import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("/Users/mac/Desktop/DEPI_Fproject/app/dt_model.pkl")

# Page Config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigate", ["Predict Churn", "Insights", "About"])

# PREDICTION SECTION
if menu == "Predict Churn":
    st.title("Customer Churn Prediction App")
    st.markdown("Use this tool to predict whether a customer will churn based on their profile.")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 30)
            support_calls = st.number_input("Support Calls", 0, 50, 2)
            payment_delay = st.number_input("Payment Delay (days)", 0, 100, 5)

        with col2:
            total_spend = st.number_input("Total Spend", 0.0, 10000.0, 1000.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
            tenure_category = st.selectbox("Tenure Category", [
                "New (≤6 months)", "Short-term (6-12 months)", 
                "Mid-term (1-3 years)", "Long-term (>3 years)"
            ])

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            "Total Spend": [total_spend],
            "Support Calls": [support_calls],
            "Age": [age],
            "Payment Delay": [payment_delay],
            "Contract Length": [contract_length],
            "Gender": [gender],
            "Tenure_Category": [tenure_category]
        })

        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is likely to stay.")

        st.markdown("#### Prediction Probability")
        st.write(f"Stay: {proba[0]:.4f} | Churn: {proba[1]:.4f}")

        # Plot probability chart
        fig, ax = plt.subplots()
        ax.bar(["Stay", "Churn"], proba, color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

# INSIGHTS SECTION
elif menu == "Insights":
    st.title("Customer Data Insights")
    st.markdown("Visual insights from customer data based on selected features.")

    df = pd.read_csv('/Users/mac/Desktop/DEPI_Fproject/app/cleaned_customer_data.csv')
    st.write("Preview of Loaded Data", df.head())

    # Optional: Search by Customer ID
    if 'CustomerID' in df.columns:
        st.subheader("Search by Customer ID")
        search_id = st.text_input("Enter Customer ID:")
        if search_id:
            result = df[df['CustomerID'].astype(str).str.contains(search_id)]
            if not result.empty:
                st.write(f"Found {len(result)} matching record(s):")
                st.dataframe(result)
            else:
                st.warning("No matching customer found.")

    # Filter by churn status
    st.subheader("Filter by Churn Status")
    churn_filter = st.selectbox("Select Churn Status", options=["All", "Stay (0)", "Churn (1)"])
    if churn_filter == "Stay (0)":
        filtered_df = df[df['Churn'] == 0]
    elif churn_filter == "Churn (1)":
        filtered_df = df[df['Churn'] == 1]
    else:
        filtered_df = df.copy()

    st.write("Filtered Data", filtered_df.head())

    # Export filtered results
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_customer_data.csv',
        mime='text/csv'
    )

    # Show histograms
    st.subheader("Distribution of Numerical Features")
    num_cols = ['Total Spend', 'Support Calls', 'Age', 'Payment Delay']
    for col in num_cols:
        st.markdown(f"**{col}**")
        fig, ax = plt.subplots()
        ax.hist(filtered_df[col], bins=20, color="skyblue", edgecolor="black")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Gender pie chart
    st.subheader("Gender Distribution")
    gender_counts = filtered_df['Gender'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"])
    ax.axis("equal")
    st.pyplot(fig)

    # Churn pie chart (fixed)
    st.subheader("Churn Distribution")
    churn_counts = filtered_df['Churn'].value_counts().sort_index()
    churn_label_map = {0: "Stay", 1: "Churn"}
    labels = [churn_label_map[i] for i in churn_counts.index]
    color_map = {0: "#66b3ff", 1: "#ff9999"}
    colors = [color_map[i] for i in churn_counts.index]

    fig, ax = plt.subplots()
    ax.pie(churn_counts, labels=labels, autopct="%1.1f%%", colors=colors)
    ax.axis("equal")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap (Numerical Features)")
    corr_matrix = filtered_df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ABOUT SECTION
else:
    st.title("About This App")
    st.markdown("""
    This app predicts customer churn using a machine learning model trained on behavioral and demographic data.

    **Model Details:**
    - Model: Decision Tree
    - Features Used: Total Spend, Support Calls, Age, Payment Delay, Contract Length, Gender, Tenure_Category
    - Accuracy: ~98% on test data
    - Preprocessing includes scaling and encoding

    Developed with Streamlit.
    """)
