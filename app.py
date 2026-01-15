import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border: 2px solid #ff4b4b;
        transform: scale(1.02);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        # Try multiple possible locations
        possible_paths = [
            (".", "."),  # Same directory
            ("./", "./"),  # Explicit current directory
            (os.getcwd(), os.getcwd()),  # Current working directory
        ]
        
        model = None
        scaler = None
        
        for model_dir, scaler_dir in possible_paths:
            try:
                model_path = os.path.join(model_dir, "fraud_detection_model.pkl")
                scaler_path = os.path.join(scaler_dir, "scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    with open(scaler_path, 'rb') as file:
                        scaler = pickle.load(file)
                    st.success(f"✅ Models loaded from: {model_dir}")
                    break
            except Exception as e:
                continue
        
        if model is None or scaler is None:
            st.error("❌ Could not load model files")
            st.info("""
            **Files Required:**
            - fraud_detection_model.pkl
            - scaler.pkl
            
            **Make sure these files are in your repository root directory**
            """)
            st.stop()
            
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

# Load models
try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

# Title
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🔒 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'>Real-time ML-powered fraud detection using XGBoost</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #666;'>Developed by Rajalekshmi Reji | Machine Learning Internship Level 3 - Challenge 2</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.markdown("### 🤖 Model Information")
    st.info("**Algorithm:** XGBoost Classifier")
    
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "99.78%")
        st.metric("Recall", "87.76%")
        st.metric("ROC-AUC", "98.36%")
    with col2:
        st.metric("Precision", "42.79%")
        st.metric("F1-Score", "57.53%")
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    st.write("• **Total Transactions:** 284,807")
    st.write("• **Legitimate:** 284,315 (99.83%)")
    st.write("• **Fraudulent:** 492 (0.17%)")
    
    st.markdown("---")
    st.markdown("### 📋 Risk Level Guide")
    st.markdown("""
    - **Prediction = 1**: 🔴 FRAUD
    - **Prediction = 0, Prob ≥ 30%**: 🟡 REVIEW
    - **Prediction = 0, Prob < 30%**: 🟢 APPROVE
    """)

# Main content
st.markdown("## 🔍 Single Transaction Analysis")

# Input section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 💳 Transaction Details")
    time_seconds = st.number_input(
        "⏱️ Time (seconds)", 
        min_value=0.0, 
        max_value=200000.0,
        value=50000.0, 
        step=1000.0
    )
    amount = st.number_input(
        "💵 Amount ($)", 
        min_value=0.0, 
        max_value=30000.0,
        value=100.0, 
        step=10.0
    )

with col2:
    st.markdown("### 🔢 Quick Presets")
    preset = st.selectbox(
        "🎯 Test Cases",
        ["Custom Values", "Known Legitimate", "Known Fraud"],
        help="Select a preset to quickly test"
    )

# PCA Features Input
st.markdown("---")
st.markdown("### 🔢 Enter V1-V28 Feature Values")

features = []

if preset == "Known Fraud":
    # Real fraud case that should trigger detection
    fraud_values = [
        -1.3598071336738,
        -0.0727811733098497,
        2.53634673796914,
        1.37815522427443,
        -0.338320769942518,
        0.462387777762292,
        0.239598554061257,
        0.0986979012610507,
        0.363786969611213,
        0.0907941719789316,
        -0.551599533260813,
        -0.617800855762348,
        -0.991389847235408,
        -0.311169353699879,
        1.46817697209427,
        -0.470400525259478,
        0.207971241929242,
        0.0257905801985591,
        0.403992960255733,
        0.251412098239705,
        -0.018306777944153,
        0.277837575558899,
        -0.110473910188767,
        0.0669280749146731,
        0.128539358273528,
        -0.189114843888824,
        0.133558376740387,
        -0.0210530534538215,
    ]
    features = fraud_values
    time_seconds = 84.0
    amount = 529.0
    
    st.warning("⚠️ Using known fraudulent transaction")
    
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.text_input(f"V{i+1}", value=f"{fraud_values[i]:.6f}", disabled=True, key=f"fraud_v{i+1}")

elif preset == "Known Legitimate":
    features = [0.0] * 28
    time_seconds = 50000.0
    amount = 100.0
    
    st.success("✅ Using known legitimate transaction")
    
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.text_input(f"V{i+1}", value="0.000000", disabled=True, key=f"legit_v{i+1}")

else:
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            feature_value = st.number_input(
                f"V{i+1}",
                value=0.0,
                step=0.1,
                format="%.6f",
                key=f"custom_v{i+1}"
            )
            features.append(feature_value)

# Analyze button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button("🔍 ANALYZE TRANSACTION", use_container_width=True)

if analyze_button:
    with st.spinner("🔄 Analyzing transaction..."):
        try:
            # Prepare input - CRITICAL: Match exact training order
            # Order must be: Time, V1-V28, Amount
            input_data = np.array([[time_seconds] + features + [amount]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            # Extract results
            prediction_class = int(prediction[0])  # 0 = Legitimate, 1 = Fraud
            fraud_probability = float(probability[0][1] * 100)
            legit_probability = float(probability[0][0] * 100)
            confidence = float(max(probability[0]) * 100)
            
            # Determine risk classification
            if prediction_class == 1:
                classification = "FRAUDULENT"
                risk_level = "🔴 HIGH RISK"
                recommendation = "DECLINE TRANSACTION"
            elif fraud_probability >= 30:
                classification = "LEGITIMATE (SUSPICIOUS)"
                risk_level = "🟠 MEDIUM RISK"
                recommendation = "MANUAL REVIEW REQUIRED"
            else:
                classification = "LEGITIMATE"
                risk_level = "🟢 LOW RISK"
                recommendation = "APPROVE TRANSACTION"
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if prediction_class == 1:
                    st.metric("🚨 Classification", "FRAUD")
                elif fraud_probability >= 30:
                    st.metric("⚠️ Classification", "SUSPICIOUS")
                else:
                    st.metric("✅ Classification", "LEGITIMATE")
            
            with col2:
                st.metric("🎲 Fraud Probability", f"{fraud_probability:.2f}%")
            
            with col3:
                st.metric("✓ Confidence", f"{confidence:.2f}%")
            
            with col4:
                st.metric("⚠️ Risk Level", risk_level)
            
            # Probability breakdown
            st.markdown("---")
            st.markdown("### 📈 Probability Breakdown")
            
            col1, col2 = st.columns(2)
            with col1:
                st.progress(fraud_probability / 100.0)
                st.write(f"**Fraud Probability:** {fraud_probability:.2f}%")
            
            with col2:
                st.progress(legit_probability / 100.0)
                st.write(f"**Legitimate Probability:** {legit_probability:.2f}%")
            
            # Detailed results
            st.markdown("---")
            
            if prediction_class == 1:
                st.error(f"""
                ### 🚨 HIGH RISK - FRAUDULENT TRANSACTION DETECTED
                
                **Risk Assessment:**
                - **Classification:** 🔴 FRAUDULENT
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Recommendation:** ⛔ **{recommendation}**
                
                **Transaction Details:**
                - 💵 Amount: ${amount:,.2f}
                - ⏱️ Time: {time_seconds:,.0f} seconds
                
                **Immediate Actions:**
                1. 🚫 DECLINE transaction
                2. 📞 Alert cardholder
                3. 🔍 Review recent history
                """)
                
            elif fraud_probability >= 30:
                st.warning(f"""
                ### ⚠️ MEDIUM RISK - MANUAL REVIEW REQUIRED
                
                **Risk Assessment:**
                - **Classification:** 🟡 {classification}
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Recommendation:** 🔍 **{recommendation}**
                
                **Actions:**
                1. 📞 Contact cardholder for verification
                2. ✅ Approve if confirmed
                3. 🚫 Decline if suspicious
                """)
                
            else:
                st.success(f"""
                ### ✅ LOW RISK - LEGITIMATE TRANSACTION
                
                **Risk Assessment:**
                - **Classification:** 🟢 {classification}
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Recommendation:** ✅ **{recommendation}**
                
                **Action:** Process normally
                """)
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.markdown("### Model Prediction")
                st.write(f"**Raw Prediction:** {prediction_class} (0=Legitimate, 1=Fraud)")
                st.write(f"**Probability:** [Legit: {legit_probability:.2f}%, Fraud: {fraud_probability:.2f}%]")
                
                summary_df = pd.DataFrame({
                    'Feature': ['Time', 'Amount'] + [f'V{i+1}' for i in range(28)],
                    'Value': [time_seconds, amount] + features
                })
                st.dataframe(summary_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"""
            ### ❌ Error During Prediction
            
            **Error:** {str(e)}
            
            **Troubleshooting:**
            1. Check all values are numeric
            2. Verify model files loaded correctly
            3. Try using a preset test case
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🔒 Credit Card Fraud Detection System</h3>
        <p><strong>Powered by XGBoost Machine Learning</strong></p>
        <p>Developed by <strong>Rajalekshmi Reji</strong></p>
        <p>Machine Learning Internship Level 3 - Challenge 2</p>
    </div>
""", unsafe_allow_html=True)