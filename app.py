import streamlit as st
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="German Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Premium Glassmorphism Theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .main .block-container {
        padding-top: 2rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2D3748;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    h1 {
        color: #4C51BF;
    }
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        color: white;
    }
    /* Improve Sidebar Navigation UI */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 12px 15px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(102, 126, 234, 0.1);
        transform: translateX(5px);
    }
    [data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] {
        font-weight: 600;
        color: #2D3748;
        font-size: 1.05rem;
    }
    /* Hide Streamlit Default Menu/Toolbar (GitHub icon, etc.) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE SETUP ---
DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            sex TEXT,
            job INTEGER,
            housing TEXT,
            saving_accounts TEXT,
            checking_account TEXT,
            credit_amount REAL,
            duration INTEGER,
            purpose TEXT,
            predicted_risk TEXT,
            probability_good REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_predictions_to_db(df, preds, probs):
    # Prepare dataframe for SQL
    save_df = df.copy()
    save_df.columns = [col.lower().replace(' ', '_') for col in save_df.columns]
    
    # Add prediction columns
    save_df['predicted_risk'] = ["Good" if p == 1 else "Bad" for p in preds]
    save_df['probability_good'] = probs
    save_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to SQLite
    conn = sqlite3.connect(DB_NAME)
    save_df.to_sql('predictions', conn, if_exists='append', index=False)
    conn.close()


# --- LOAD MODELS & PREPROCESSORS ---
@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    encoders_path = os.path.join("models", "label_encoders.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
        st.error("Model assets not found. Please run train.py first.")
        st.stop()
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    return model, scaler, label_encoders

model, scaler, label_encoders = load_assets()

# --- PREDICTION FUNCTION ---
def predict_risk(data_df):
    """Preprocess the dataframe and return predictions"""
    df_pred = data_df.copy()
    
    # Handle missing values
    if 'Saving accounts' in df_pred.columns:
        df_pred['Saving accounts'] = df_pred['Saving accounts'].fillna('unknown')
    if 'Checking account' in df_pred.columns:
        df_pred['Checking account'] = df_pred['Checking account'].fillna('unknown')
        
    # Ensure correct column order
    features = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    
    # Encode categorical variables
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        if col in df_pred.columns:
            le = label_encoders[col]
            # Handle unseen labels by assigning a default or the first class
            df_pred[col] = df_pred[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_pred[col] = le.transform(df_pred[col])
            
    df_pred = df_pred[features] # Reorder
    
    # Scale numerical variables
    X_scaled = scaler.transform(df_pred)
    
    # Predict
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1] # Probability of 'Good' risk (1)
    
    return preds, probs

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 🏦 Navigation")
st.sidebar.markdown("Select a mode to continue:")
app_mode = st.sidebar.radio("", ["👤 Single Applicant", "📁 Batch Prediction", "📜 Prediction History"], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown("""
<div style="background-color: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea; padding: 15px; border-radius: 8px; margin-top: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
    <h4 style="color: #4C51BF; margin-top: 0; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.1rem;">ℹ️ About</h4>
    <p style="font-size: 0.9rem; color: #4A5568; margin-bottom: 0; line-height: 1.4;">
        This dashboard uses an optimized <strong>XGBoost</strong> model to predict credit risk. It evaluates key financial and personal indicators to provide an instant, accurate risk assessment.
    </p>
</div>
""", unsafe_allow_html=True)

# --- MAIN APP ---
st.title("German Credit Risk Predictor")
st.markdown("Use this dashboard to predict whether an applicant is a **Good** or **Bad** credit risk based on their profile.")
st.divider()

if app_mode == "👤 Single Applicant":
    st.subheader("👤 Single Applicant Prediction")
    st.markdown("Enter the applicant's details below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job Type (0: unskilled, 1: skilled, 2: highly skilled, 3: management)", [0, 1, 2, 3], index=2)
        
    with col2:
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
        
    with col3:
        credit_amount = st.number_input("Credit Amount (DM)", min_value=100, max_value=20000, value=3000)
        duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=24)
        purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"])

    if st.button("Predict Risk 🚀", use_container_width=True):
        input_data = {
            'Age': [age],
            'Sex': [sex],
            'Job': [job],
            'Housing': [housing],
            'Saving accounts': [saving_accounts],
            'Checking account': [checking_account],
            'Credit amount': [credit_amount],
            'Duration': [duration],
            'Purpose': [purpose]
        }
        input_df = pd.DataFrame(input_data)
        
        with st.spinner("Analyzing applicant profile..."):
            pred, prob = predict_risk(input_df)
            save_predictions_to_db(input_df, pred, prob)
            
        st.divider()
        st.subheader("Prediction Result")
        
        if pred[0] == 1:
            st.success(f"✅ **GOOD CREDIT RISK** (Confidence: {prob[0]:.1%})")
            st.balloons()
        else:
            st.error(f"❌ **BAD CREDIT RISK** (Confidence: {1 - prob[0]:.1%})")

elif app_mode == "📁 Batch Prediction":
    st.subheader("📁 Batch Upload Prediction")
    st.markdown("Upload a CSV file containing applicant data. The CSV should have the same columns as the German Credit dataset.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())
            
            if st.button("Run Batch Prediction 📊"):
                with st.spinner("Processing batch predictions..."):
                    preds, probs = predict_risk(df_batch)
                    
                    df_batch['Predicted_Risk'] = ["Good" if p == 1 else "Bad" for p in preds]
                    df_batch['Probability_Good'] = probs
                    
                    save_predictions_to_db(df_batch.drop(columns=['Predicted_Risk', 'Probability_Good']), preds, probs)
                    
                st.success("Batch prediction completed successfully and saved to Database!")
                st.dataframe(df_batch[['Predicted_Risk', 'Probability_Good'] + list(df_batch.columns[:-2])])
                
                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV 📥",
                    data=csv,
                    file_name='credit_risk_predictions.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please ensure your CSV has the following columns: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose")

elif app_mode == "📜 Prediction History":
    st.subheader("📜 Database Prediction History")
    st.markdown("View all past predictions stored in the database.")
    
    conn = sqlite3.connect(DB_NAME)
    try:
        history_df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        if len(history_df) == 0:
            st.info("No predictions found in the database yet. Run a prediction to see history here.")
        else:
            # Metrics
            total_preds = len(history_df)
            good_preds = len(history_df[history_df['predicted_risk'] == 'Good'])
            bad_preds = total_preds - good_preds
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Predictions", total_preds)
            c2.metric("Good Risk Predicted", good_preds)
            c3.metric("Bad Risk Predicted", bad_preds)
            
            st.dataframe(history_df, use_container_width=True)
            
            st.download_button(
                label="Export History to CSV 📥",
                data=history_df.to_csv(index=False).encode('utf-8'),
                file_name='full_prediction_history.csv',
                mime='text/csv',
            )
            
            st.divider()
            if st.button("Clear History 🗑️"):
                c = conn.cursor()
                c.execute("DELETE FROM predictions")
                conn.commit()
                st.rerun()
                
    except Exception as e:
        st.error(f"Could not load prediction history: {e}")
    finally:
        conn.close()
