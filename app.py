# app.py — Complete Single File App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Loads trained ML models.
 
# ─── Page Config ───
st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide" # Makes the app use the entire screen.
)

# ─── Load Data & Models ───
@st.cache_data  # Don't read the CSV every time.
def load_data():
    df_encoded = pd.read_csv(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\telco_churn_final_complete.csv")
    df_original = pd.read_csv(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\telco_churn_featured.csv") # Used for graphs.
    df_original['Churn'] = df_original['Churn'].replace({'Yes': 1, 'No': 0}) # because mathematical calculations require numbers.
    df_original['Churn'] = pd.to_numeric(df_original['Churn'],
                            errors='coerce').fillna(0).astype(int)
    return df_encoded, df_original

@st.cache_resource
def load_models():
    clf = joblib.load(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\churn_classifier.pkl")
    reg = joblib.load(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\tenure_regressor.pkl")
    kmeans = joblib.load(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\kmeans_cluster.pkl")
    scaler = joblib.load(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\cluster_scaler.pkl")
    clf_features = joblib.load(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\feature_columns.pkl")
    reg_features = joblib.load(r"C:\Users\ASUS\Python Jupyter\Customer_Intelligence_System\regression_features.pkl")
    return clf, reg, kmeans, scaler, clf_features, reg_features

df, df_viz = load_data()
# df      = df_encoded
# df_viz  = df_original
clf, reg, kmeans, scaler, clf_features, reg_features = load_models()

# ─── Sidebar Navigation ───
st.sidebar.title("📊 Customer Intelligence")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📈 Customer Overview",
    "🔮 Churn Predictor",
    "📅 Tenure Estimator",
    "👥 Customer Segments"
])

# ════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════
if page == "🏠 Home":
    st.title("📊 Customer Intelligence System")
    st.subheader("Telco Customer Analysis — ML Powered Dashboard")

    st.markdown("""
    ### Welcome!
    Use the sidebar to navigate between pages:
    - 📈 **Customer Overview** — EDA & Business Insights
    - 🔮 **Churn Predictor** — Predict if a customer will churn
    - 📅 **Tenure Estimator** — Predict how long a customer will stay
    - 👥 **Customer Segments** — Explore customer clusters
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df_viz):,}")
    col2.metric("Churn Rate", f"{df_viz['Churn'].mean()*100:.1f}%")
    col3.metric("Avg Monthly Charges", f"${df_viz['MonthlyCharges'].mean():.2f}")
    col4.metric("Avg Tenure", f"{df_viz['tenure'].mean():.1f} months")

# ════════════════════════════════════════
# PAGE 2 — CUSTOMER OVERVIEW
# ════════════════════════════════════════
elif page == "📈 Customer Overview":
    st.title("📈 Customer Overview")
    st.markdown("EDA & Business Insights")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df_viz):,}")
    col2.metric("Churned", f"{df_viz['Churn'].sum():,}")
    col3.metric("Churn Rate", f"{df_viz['Churn'].mean()*100:.1f}%")
    col4.metric("Revenue at Risk",
                f"${df_viz[df_viz['Churn']==1]['MonthlyCharges'].sum():,.0f}/mo")

    st.divider()

    # Row 1 — Churn Analysis
    st.subheader("Churn Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        contract_churn = df_viz.groupby(['Contract', 'Churn']).size().unstack()
        contract_churn.plot(kind='bar', ax=ax,
                            color=['#66c2a5', '#fc8d62'],
                            edgecolor='white')
        ax.set_title('Churn by Contract Type')
        ax.set_xlabel('')
        ax.legend(labels=['Not Churned', 'Churned'])
        plt.xticks(rotation=360)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        order = ['New', 'Mid', 'Loyal', 'Champion']
        churn_tenure = df_viz.groupby('tenure_group')['Churn'].mean() * 100
        churn_tenure = churn_tenure.reindex(order)
        churn_tenure.plot(kind='bar', ax=ax,
                          color='steelblue', edgecolor='white')
        ax.set_title('Churn Rate by Tenure Group (%)')
        ax.set_ylabel('Churn Rate (%)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # Row 2 — Revenue & Services
    st.subheader("Revenue & Services")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        not_churned = df_viz[df_viz['Churn'] == 0]['MonthlyCharges']
        churned = df_viz[df_viz['Churn'] == 1]['MonthlyCharges']
        ax.hist(not_churned, bins=30, alpha=0.6,
                color='#66c2a5', label='Not Churned')
        ax.hist(churned, bins=30, alpha=0.6,
                color='#fc8d62', label='Churned')
        ax.set_title('Monthly Charges Distribution')
        ax.set_xlabel('Monthly Charges')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        not_churned_s = df_viz[df_viz['Churn'] == 0]['services_count']
        churned_s = df_viz[df_viz['Churn'] == 1]['services_count']
        ax.boxplot([not_churned_s, churned_s],
                   labels=['Not Churned', 'Churned'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#66c2a5'),
                   medianprops=dict(color='black'))
        ax.set_title('Services Count by Churn')
        ax.set_ylabel('Services Count')
        plt.tight_layout()
        st.pyplot(fig)

# ════════════════════════════════════════
# PAGE 3 — CHURN PREDICTOR
# ════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.title("🔮 Churn Predictor")
    st.markdown("Enter customer details to predict churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"])

    with col2:
        st.subheader("Services")
        security = st.selectbox("Online Security", ["No", "Yes"])
        backup = st.selectbox("Online Backup", ["No", "Yes"])
        protection = st.selectbox("Device Protection", ["No", "Yes"])
        techsupport = st.selectbox("Tech Support", ["No", "Yes"])
        streamingtv = st.selectbox("Streaming TV", ["No", "Yes"])
        streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes"])

    with col3:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        contract = st.selectbox("Contract",
                                ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
        total = st.number_input("Total Charges", 0.0, 9000.0,
                                float(monthly * tenure))
    
    if st.button("🔮 Predict Churn", use_container_width=True):
        # Feature Engineering
        services = [phone, multiple, security, backup,
                    protection, techsupport, streamingtv, streamingmovies]
        services_count = sum([1 for s in services if s == "Yes"])
        avg_daily = round(monthly / 30, 2)
        charge_per_ten = round(total / max(tenure, 1), 2)

        if tenure <= 12:
            tenure_group = "New"
        elif tenure <= 24:
            tenure_group = "Mid"
        elif tenure <= 48:
            tenure_group = "Loyal"
        else:
            tenure_group = "Champion"

        input_dict = {
            'gender': 1 if gender == "Male" else 0,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone == "Yes" else 0,
            'MultipleLines': 1 if multiple == "Yes" else 0,
            'OnlineSecurity': 1 if security == "Yes" else 0,
            'OnlineBackup': 1 if backup == "Yes" else 0,
            'DeviceProtection': 1 if protection == "Yes" else 0,
            'TechSupport': 1 if techsupport == "Yes" else 0,
            'StreamingTV': 1 if streamingtv == "Yes" else 0,
            'StreamingMovies': 1 if streamingmovies == "Yes" else 0,
            'PaperlessBilling': 1 if paperless == "Yes" else 0,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'avg_daily_charge': avg_daily,
            'services_count': services_count,
            'charge_per_tenure': charge_per_ten,
            'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
            'InternetService_No': 1 if internet == "No" else 0,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
            'tenure_group_Loyal': 1 if tenure_group == "Loyal" else 0,
            'tenure_group_Mid': 1 if tenure_group == "Mid" else 0,
            'tenure_group_New': 1 if tenure_group == "New" else 0,
        }

        input_df = pd.DataFrame([input_dict])[clf_features]
        prob = clf.predict_proba(input_df)[0][1]
        pred = clf.predict(input_df)[0]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ High Churn Risk — Probability: {prob*100:.1f}%")
        else:
            st.success(f"✅ Low Churn Risk — Probability: {prob*100:.1f}%")

        st.progress(float(prob))
        st.caption(f"Churn probability: {prob*100:.1f}%")

# ════════════════════════════════════════
# PAGE 4 — TENURE ESTIMATOR
# ════════════════════════════════════════
elif page == "📅 Tenure Estimator":
    st.title("📅 Tenure Estimator")
    st.markdown("Predict how long a customer will stay.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Profile")
        gender_r = st.selectbox("Gender", ["Male", "Female"])
        senior_r = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner_r = st.selectbox("Partner", ["Yes", "No"])
        dependents_r = st.selectbox("Dependents", ["No", "Yes"])
        phone_r = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_r = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_r = st.selectbox("Internet Service",
                                  ["DSL", "Fiber optic", "No"])

    with col2:
        st.subheader("Account Details")
        security_r = st.selectbox("Online Security", ["No", "Yes"])
        backup_r = st.selectbox("Online Backup", ["No", "Yes"])
        protection_r = st.selectbox("Device Protection", ["No", "Yes"])
        techsupport_r = st.selectbox("Tech Support", ["No", "Yes"])
        streamingtv_r = st.selectbox("Streaming TV", ["No", "Yes"])
        streamingmovies_r = st.selectbox("Streaming Movies", ["No", "Yes"])
        contract_r = st.selectbox("Contract",
                                  ["Month-to-month", "One year", "Two year"])
        paperless_r = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_r = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])

    if st.button("📅 Estimate Tenure", use_container_width=True):
        services_r = [phone_r, multiple_r, security_r, backup_r,
                      protection_r, techsupport_r,
                      streamingtv_r, streamingmovies_r]
        services_count_r = sum([1 for s in services_r if s == "Yes"])

        input_dict_r = {
            'gender': 1 if gender_r == "Male" else 0,
            'SeniorCitizen': 1 if senior_r == "Yes" else 0,
            'Partner': 1 if partner_r == "Yes" else 0,
            'Dependents': 1 if dependents_r == "Yes" else 0,
            'PhoneService': 1 if phone_r == "Yes" else 0,
            'MultipleLines': 1 if multiple_r == "Yes" else 0,
            'OnlineSecurity': 1 if security_r == "Yes" else 0,
            'OnlineBackup': 1 if backup_r == "Yes" else 0,
            'DeviceProtection': 1 if protection_r == "Yes" else 0,
            'TechSupport': 1 if techsupport_r == "Yes" else 0,
            'StreamingTV': 1 if streamingtv_r == "Yes" else 0,
            'StreamingMovies': 1 if streamingmovies_r == "Yes" else 0,
            'PaperlessBilling': 1 if paperless_r == "Yes" else 0,
            'services_count': services_count_r,
            'InternetService_Fiber optic': 1 if internet_r == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_r == "No" else 0,
            'Contract_One year': 1 if contract_r == "One year" else 0,
            'Contract_Two year': 1 if contract_r == "Two year" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_r == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment_r == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment_r == "Mailed check" else 0,
        }

        input_df_r = pd.DataFrame([input_dict_r])[reg_features]
        predicted_tenure = reg.predict(input_df_r)[0]
        predicted_tenure = max(1, round(predicted_tenure))

        st.divider()
        st.success(f"📅 Estimated Tenure: **{predicted_tenure} months**")
        st.info(f"This customer is expected to stay approximately "
                f"**{predicted_tenure // 12} years and "
                f"{predicted_tenure % 12} months**")
        st.progress(min(predicted_tenure / 72, 1.0))

# ════════════════════════════════════════
# PAGE 5 — CUSTOMER SEGMENTS
# ════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")
    st.markdown("K-Means Clustering — 3 Customer Segments")

    segment_names = {
        0: "Basic Users",
        1: "At Risk High Spenders",
        2: "Champion Customers"
    }
    df['Segment'] = df['Cluster'].map(segment_names)

    col1, col2, col3 = st.columns(3)
    for i, (cluster, name) in enumerate(segment_names.items()):
        subset = df[df['Cluster'] == cluster]
        [col1, col2, col3][i].metric(
            name,
            f"{len(subset):,} customers",
            f"{subset['Churn'].mean()*100:.1f}% churn"
        )

    st.divider()

    st.subheader("Segment Profiles")
    profile = df.groupby('Segment')[
        ['tenure', 'MonthlyCharges', 'services_count', 'Churn']
    ].mean().round(2)
    profile.columns = ['Avg Tenure', 'Avg Monthly Charges',
                       'Avg Services', 'Churn Rate']
    profile['Churn Rate'] = (profile['Churn Rate'] * 100).round(1)
    st.dataframe(profile, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        df['Segment'].value_counts().plot(
            kind='bar', ax=ax, color=colors, edgecolor='white')
        ax.set_title('Customer Count by Segment')
        ax.set_xlabel('')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_seg = df.groupby('Segment')['Churn'].mean() * 100
        churn_by_seg.plot(kind='bar', ax=ax,
                          color=colors, edgecolor='white')
        ax.set_title('Churn Rate by Segment (%)')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_xlabel('')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)