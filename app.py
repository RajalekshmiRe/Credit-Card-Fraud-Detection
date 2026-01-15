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

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load trained model and scaler"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model
        model_path = os.path.join(current_dir, "fraud_detection_model.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please ensure 'fraud_detection_model.pkl' is in the same folder as app.py")
            st.stop()
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Load scaler
        scaler_path = os.path.join(current_dir, "scaler.pkl")
        
        if not os.path.exists(scaler_path):
            st.error(f"❌ Scaler file not found at: {scaler_path}")
            st.info("Please ensure 'scaler.pkl' is in the same folder as app.py")
            st.stop()
            
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

# Load models
model, scaler = load_models()

# Title
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🔒 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'>Real-time ML-powered fraud detection using XGBoost</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #666;'>Developed by Rajalekshmi Reji | ML Internship Level 3 - Challenge 2</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
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

# Main content
st.markdown("## 🔍 Single Transaction Analysis")

# DEFINE THE KNOWN VALUES FIRST - SINGLE SOURCE OF TRUTH
# ✅ THESE ARE THE EXACT VALUES FROM YOUR COLAB THAT WORK!
KNOWN_FRAUD_VALUES = {
    'time': 84.0,
    'amount': 529.0,
    'features': [
        -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443,
        -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507,
        0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348,
        -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
        0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705,
        -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731,
        0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215
    ]
}

KNOWN_LEGIT_VALUES = {
    'time': 150.0,
    'amount': 50.0,
    'features': [
        1.191857, 0.266151, 0.166480, 0.448154, 0.060018,
        -0.082361, -0.078803, 0.085102, -0.255425, -0.166974,
        1.612727, 1.065235, 0.489095, -0.143772, 0.635558,
        0.463917, -0.114805, -0.183361, -0.145783, -0.069083,
        -0.225775, -0.638672, 0.101288, -0.339846, 0.167170,
        0.125895, -0.008983, 0.014724
    ]
}

col1, col2 = st.columns([1, 1])

with col2:
    st.markdown("### 🎯 Quick Presets")
    preset = st.selectbox(
        "Select Test Case",
        ["Custom Values", "Known Fraudulent Transaction", "Known Legitimate Transaction"],
        key="preset_selector"
    )

# Initialize values based on preset
if preset == "Known Fraudulent Transaction":
    initial_time = KNOWN_FRAUD_VALUES['time']
    initial_amount = KNOWN_FRAUD_VALUES['amount']
    features = KNOWN_FRAUD_VALUES['features']
elif preset == "Known Legitimate Transaction":
    initial_time = KNOWN_LEGIT_VALUES['time']
    initial_amount = KNOWN_LEGIT_VALUES['amount']
    features = KNOWN_LEGIT_VALUES['features']
else:
    initial_time = 50000.0
    initial_amount = 100.0
    features = [0.0] * 28

with col1:
    st.markdown("### 💳 Transaction Details")
    time_seconds = st.number_input(
        "⏱️ Time (seconds)", 
        min_value=0.0, 
        value=initial_time,
        step=1000.0,
        key="time_input"
    )
    amount = st.number_input(
        "💵 Amount ($)", 
        min_value=0.0, 
        value=initial_amount,
        step=10.0,
        key="amount_input"
    )

# PCA Features
st.markdown("---")
st.markdown("### 🔢 Enter V1-V28 Feature Values")

# Display features based on preset
if preset == "Known Fraudulent Transaction":
    st.warning("⚠️ Using known fraudulent transaction")
    
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.text_input(
                f"V{i+1}", 
                value=f"{features[i]:.6f}", 
                disabled=True, 
                key=f"fraud_v{i+1}"
            )

elif preset == "Known Legitimate Transaction":
    st.success("✅ Using known legitimate transaction")
    
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.text_input(
                f"V{i+1}", 
                value=f"{features[i]:.6f}", 
                disabled=True, 
                key=f"legit_v{i+1}"
            )

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
    analyze = st.button("🔍 ANALYZE TRANSACTION", use_container_width=True)

if analyze:
    with st.spinner("🔄 Analyzing..."):
        try:
            # Prepare input data - USE THE VALUES FROM THE PRESET OR INPUTS
            input_data = np.array([[time_seconds] + features + [amount]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            # Extract values
            prediction_class = int(prediction[0])
            fraud_probability = float(probability[0][1] * 100)
            legit_probability = float(probability[0][0] * 100)
            confidence = float(max(probability[0]) * 100)
            
            # Determine classification
            if prediction_class == 1:
                classification = "FRAUDULENT"
                risk_level = "🔴 HIGH RISK"
                recommendation = "DECLINE TRANSACTION"
            elif fraud_probability >= 30:
                classification = "SUSPICIOUS"
                risk_level = "🟠 MEDIUM RISK"
                recommendation = "MANUAL REVIEW"
            else:
                classification = "LEGITIMATE"
                risk_level = "🟢 LOW RISK"
                recommendation = "APPROVE"
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚨 Classification", classification)
            with col2:
                st.metric("🎲 Fraud Probability", f"{fraud_probability:.2f}%")
            with col3:
                st.metric("✓ Confidence", f"{confidence:.2f}%")
            with col4:
                st.metric("⚠️ Risk Level", risk_level)
            
            st.markdown("---")
            st.markdown("### 📈 Probability Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.progress(fraud_probability / 100.0)
                st.write(f"**Fraud:** {fraud_probability:.2f}%")
            with col2:
                st.progress(legit_probability / 100.0)
                st.write(f"**Legitimate:** {legit_probability:.2f}%")
            
            st.markdown("---")
            
            if prediction_class == 1:
                st.error(f"""
                ### 🚨 HIGH RISK - FRAUDULENT TRANSACTION
                
                **Risk Assessment:**
                - **Classification:** 🔴 FRAUDULENT
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Confidence:** {confidence:.2f}%
                - **Recommendation:** ⛔ **{recommendation}**
                
                **Transaction Details:**
                - 💵 Amount: ${amount:,.2f}
                - ⏱️ Time: {time_seconds:,.0f} seconds
                
                **Actions Required:**
                1. 🚫 DECLINE transaction
                2. 📞 Alert cardholder
                3. 🔍 Review recent history
                4. 🔒 Consider card suspension
                """)
                
            elif fraud_probability >= 30:
                st.warning(f"""
                ### ⚠️ MEDIUM RISK - MANUAL REVIEW
                
                **Risk Assessment:**
                - **Classification:** 🟡 {classification}
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Recommendation:** 🔍 **{recommendation}**
                
                **Actions:**
                1. 📞 Contact cardholder
                2. ✅ Approve if verified
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
                st.write(f"**Raw Prediction:** {prediction_class} (0=Legit, 1=Fraud)")
                st.write(f"**Probabilities:** [Legit: {legit_probability:.2f}%, Fraud: {fraud_probability:.2f}%]")
                
                st.markdown("### Input Features")
                summary_df = pd.DataFrame({
                    'Feature': ['Time', 'Amount'] + [f'V{i+1}' for i in range(28)],
                    'Value': [time_seconds, amount] + features
                })
                st.dataframe(summary_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🔒 Credit Card Fraud Detection System</h3>
        <p><strong>Powered by XGBoost Machine Learning</strong></p>
        <p>Accuracy: 99.78% | Recall: 87.76%</p>
        <p>Developed by <strong>Rajalekshmi Reji</strong></p>
        <p>ML Internship Level 3 - Challenge 2 | Certify Technology</p>
    </div>
""", unsafe_allow_html=True)