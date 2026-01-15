import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
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
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8b8b 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        border-radius: 10px;
    }
    
    /* Custom info cards */
    .info-card {
        background: linear-gradient(135deg, #1e2530 0%, #252d3d 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #ff4b4b;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .risk-indicator {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff4b4b 0%, #c92a2a 100%);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa94d 0%, #fd7e14 100%);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(current_dir, "fraud_detection_model.pkl")
        scaler_path = os.path.join(current_dir, "scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("❌ Model or scaler file not found in the app directory!")
            st.stop()
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {str(e)}")
        st.stop()

model, scaler = load_models()

# Header Section
st.markdown("<h1 style='text-align: center; color: #ff4b4b; margin-bottom: 0;'>🔒 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888; margin-top: 0.5rem;'>Real-time ML-powered fraud detection using XGBoost</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.95rem; color: #666; margin-bottom: 2rem;'>Developed by <strong>Rajalekshmi Reji</strong> | ML Internship Level 3 - Challenge 2 | Certify Technology</p>", unsafe_allow_html=True)
st.markdown("---")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.info("**Algorithm:** XGBoost Classifier")
    
    st.markdown("### 📊 Performance Metrics")
    st.markdown("---")
    
    # Metrics with better organization
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✓ Accuracy", "99.78%", delta="High", delta_color="normal")
        st.metric("🎯 Recall", "87.76%", delta="Good", delta_color="normal")
        st.metric("📈 ROC-AUC", "98.36%", delta="Excellent", delta_color="normal")
    with col2:
        st.metric("🔍 Precision", "42.79%", delta="Moderate", delta_color="off")
        st.metric("⚖️ F1-Score", "57.53%", delta="Balanced", delta_color="normal")
        st.metric("🎲 Model", "XGB", delta="v2.0", delta_color="off")
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    st.info("""
    **Total Transactions:** 284,807
    
    **Legitimate:** 284,315 (99.83%)
    
    **Fraudulent:** 492 (0.17%)
    
    **Imbalance Ratio:** 1:578
    """)
    
    st.markdown("---")
    st.markdown("### 🛡️ Model Features")
    st.success("""
    ✓ Real-time Detection
    
    ✓ High Accuracy (99.78%)
    
    ✓ SMOTE Balanced
    
    ✓ Production Ready
    
    ✓ Explainable AI
    """)
    
    st.markdown("---")
    st.markdown("### 🔐 Security Level")
    st.error("**ENTERPRISE GRADE**")
    st.caption("Trained on 284K+ transactions")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    with st.expander("📖 How it works"):
        st.write("""
        This system uses:
        - **XGBoost** algorithm
        - **PCA features** (V1-V28)
        - **SMOTE** for balancing
        - **StandardScaler** normalization
        
        It analyzes transaction patterns to detect fraud in real-time.
        """)

# Fraud values from your working app
KNOWN_FRAUD_VALUES = {
    'time': 406.0,
    'amount': 373.0,
    'features': [
        -2.312227, 1.951992, -1.609851, 3.997906, -0.522188,
        -1.426545, -2.537387, 1.391657, -2.770089, -2.772272,
        3.202033, -2.899907, -0.595222, -4.289254, 0.389724,
        -1.140747, -2.830056, -0.016822, 0.416956, 0.126911,
        0.517232, -0.035049, -0.465211, 0.320198, 0.044519,
        0.177840, 0.261145, -0.143276
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

# Main Content Area
st.markdown("## 💳 Transaction Input")

# Preset and Transaction Details
col_preset, col_time, col_amount = st.columns([2, 1, 1])

with col_preset:
    preset = st.selectbox(
        "📋 Select Test Case",
        ["Custom Values", "Known Fraudulent Transaction", "Known Legitimate Transaction"],
        help="Choose a preset transaction or enter custom values"
    )

# Initialize values based on preset
if preset == "Known Fraudulent Transaction":
    time_seconds = KNOWN_FRAUD_VALUES['time']
    amount = KNOWN_FRAUD_VALUES['amount']
    features = KNOWN_FRAUD_VALUES['features'].copy()
elif preset == "Known Legitimate Transaction":
    time_seconds = KNOWN_LEGIT_VALUES['time']
    amount = KNOWN_LEGIT_VALUES['amount']
    features = KNOWN_LEGIT_VALUES['features'].copy()
else:
    time_seconds = 50000.0
    amount = 100.0
    features = [0.0]*28

with col_time:
    time_seconds = st.number_input("⏱️ Time (seconds)", value=float(time_seconds), step=1000.0, help="Time elapsed since first transaction")

with col_amount:
    amount = st.number_input("💵 Amount ($)", value=float(amount), step=10.0, help="Transaction amount in dollars")

# Feature inputs section
st.markdown("### 🔢 PCA Feature Values (V1-V28)")
st.caption("These are PCA-transformed features from the original transaction data")

cols = st.columns(4)
if preset in ["Known Fraudulent Transaction", "Known Legitimate Transaction"]:
    for i in range(28):
        with cols[i % 4]:
            st.text_input(f"V{i+1}", value=f"{features[i]:.6f}", disabled=True, key=f"v{i+1}")
else:
    features = []
    for i in range(28):
        with cols[i % 4]:
            val = st.number_input(f"V{i+1}", value=0.0, step=0.1, format="%.6f", key=f"v{i+1}_input")
            features.append(val)

# Centered Analyze Button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 ANALYZE TRANSACTION", use_container_width=True)

# Analysis Results
if analyze_button:
    with st.spinner("🔄 Analyzing transaction patterns..."):
        try:
            # Prepare input in correct order: [Time, V1-V28, Amount]
            input_data = np.array([[time_seconds] + features + [amount]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prob = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]
            
            # Convert numpy float32 to Python float
            fraud_prob = float(prob[1]) * 100
            legit_prob = float(prob[0]) * 100
            confidence = max(fraud_prob, legit_prob)
            
            # Classification logic
            if fraud_prob >= 50:
                classification = "FRAUDULENT"
                risk_level = "🔴 HIGH RISK"
                risk_class = "risk-high"
                recommendation = "DECLINE TRANSACTION"
                rec_icon = "⛔"
                action_color = "error"
            elif fraud_prob >= 20:
                classification = "SUSPICIOUS"
                risk_level = "🟠 MEDIUM RISK"
                risk_class = "risk-medium"
                recommendation = "MANUAL REVIEW REQUIRED"
                rec_icon = "⚠️"
                action_color = "warning"
            elif fraud_prob >= 5:
                classification = "REVIEW NEEDED"
                risk_level = "🟡 LOW-MEDIUM RISK"
                risk_class = "risk-medium"
                recommendation = "AUTOMATED REVIEW"
                rec_icon = "ℹ️"
                action_color = "info"
            else:
                classification = "LEGITIMATE"
                risk_level = "🟢 LOW RISK"
                risk_class = "risk-low"
                recommendation = "APPROVE TRANSACTION"
                rec_icon = "✅"
                action_color = "success"
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Main metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🚨 Classification", classification)
            c2.metric("🎲 Fraud Probability", f"{fraud_prob:.2f}%")
            c3.metric("✓ Confidence", f"{confidence:.2f}%")
            c4.metric("⚠️ Risk Level", risk_level)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk Indicator Visual
            st.markdown(f"""
                <div class='risk-indicator {risk_class}'>
                    <h3 style='margin: 0;'>{rec_icon} {recommendation}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability bars
            st.markdown("### 📈 Probability Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Legitimate Probability**")
                st.progress(float(legit_prob / 100.0))
                st.caption(f"🟢 {legit_prob:.2f}%")
                
            with col2:
                st.markdown(f"**Fraud Probability**")
                st.progress(float(fraud_prob / 100.0))
                st.caption(f"🔴 {fraud_prob:.2f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Enhanced Recommendation Section
            st.markdown("### 💡 Detailed Analysis & Recommendations")
            
            if classification == "FRAUDULENT":
                st.error(f"### {rec_icon} {recommendation}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🚨 Risk Indicators:**")
                    st.write("• Fraud probability exceeds 50% threshold")
                    st.write(f"• Transaction amount: ${amount:.2f}")
                    st.write(f"• Confidence level: {confidence:.2f}%")
                    st.write("• Pattern matches known fraud signatures")
                
                with col2:
                    st.markdown("**📋 Recommended Actions:**")
                    st.write("1. **Immediately decline** this transaction")
                    st.write("2. **Contact cardholder** for verification")
                    st.write("3. **Flag account** for monitoring")
                    st.write("4. **Generate incident report**")
                
                st.warning("⚠️ **Alert:** This transaction shows strong indicators of fraudulent activity. Immediate action required.")
                
            elif classification == "SUSPICIOUS":
                st.warning(f"### {rec_icon} {recommendation}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**⚠️ Suspicious Indicators:**")
                    st.write("• Fraud probability between 20-50%")
                    st.write(f"• Transaction amount: ${amount:.2f}")
                    st.write(f"• Confidence level: {confidence:.2f}%")
                    st.write("• Requires human verification")
                
                with col2:
                    st.markdown("**📋 Recommended Actions:**")
                    st.write("1. **Hold transaction** temporarily")
                    st.write("2. **Route to fraud analyst** for review")
                    st.write("3. **Request additional verification**")
                    st.write("4. **Monitor cardholder activity**")
                
                st.info("ℹ️ **Note:** This transaction requires manual verification by a fraud analyst before processing.")
                
            elif classification == "REVIEW NEEDED":
                st.info(f"### {rec_icon} {recommendation}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**ℹ️ Review Indicators:**")
                    st.write("• Fraud probability between 5-20%")
                    st.write(f"• Transaction amount: ${amount:.2f}")
                    st.write(f"• Confidence level: {confidence:.2f}%")
                    st.write("• Low-risk but flagged for review")
                
                with col2:
                    st.markdown("**📋 Recommended Actions:**")
                    st.write("1. **Process with monitoring**")
                    st.write("2. **Add to automated review queue**")
                    st.write("3. **Track for pattern analysis**")
                    st.write("4. **No immediate action required**")
                
                st.info("ℹ️ **Note:** This transaction will be automatically flagged for routine review.")
                
            else:
                st.success(f"### {rec_icon} {recommendation}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Safe Transaction Indicators:**")
                    st.write("• Fraud probability below 5%")
                    st.write(f"• Transaction amount: ${amount:.2f}")
                    st.write(f"• Confidence level: {confidence:.2f}%")
                    st.write("• Matches legitimate patterns")
                
                with col2:
                    st.markdown("**📋 Processing Status:**")
                    st.write("1. **✓ Safe to approve** this transaction")
                    st.write("2. **✓ No additional verification** needed")
                    st.write("3. **✓ Standard processing** can proceed")
                    st.write("4. **✓ No alerts** generated")
                
                st.success("✅ **Status:** This transaction appears legitimate and can proceed normally.")
            
            # Transaction Summary Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📄 Transaction Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.markdown("**Transaction Details:**")
                st.write(f"• Time: {time_seconds:.0f} seconds")
                st.write(f"• Amount: ${amount:.2f}")
                st.write(f"• Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            with summary_col2:
                st.markdown("**Risk Assessment:**")
                st.write(f"• Classification: {classification}")
                st.write(f"• Risk Level: {risk_level}")
                st.write(f"• Decision: {recommendation}")
            
            with summary_col3:
                st.markdown("**Model Confidence:**")
                st.write(f"• Fraud Score: {fraud_prob:.2f}%")
                st.write(f"• Legitimate Score: {legit_prob:.2f}%")
                st.write(f"• Overall Confidence: {confidence:.2f}%")
            
            # Technical Details Expander
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔧 Technical Details & Feature Values"):
                st.markdown("#### Raw Model Probabilities")
                col1, col2 = st.columns(2)
                col1.metric("Legitimate (Class 0)", f"{float(prob[0]):.6f}", f"{legit_prob:.2f}%")
                col2.metric("Fraudulent (Class 1)", f"{float(prob[1]):.6f}", f"{fraud_prob:.2f}%")
                
                st.markdown("---")
                st.markdown("#### Model Information")
                st.code("Algorithm: XGBoost Classifier\nFeature Order: Time → V1-V28 → Amount\nScaling: StandardScaler", language="text")
                
                st.markdown("---")
                st.markdown("#### Complete Input Values")
                summary_df = pd.DataFrame({
                    "Feature": ["Time"] + [f"V{i+1}" for i in range(28)] + ["Amount"],
                    "Value": [time_seconds] + features + [amount]
                })
                st.dataframe(summary_df, use_container_width=True, height=400)
            
        except Exception as e:
            st.error(f"❌ **Error during analysis:** {str(e)}")
            with st.expander("🐛 Debug Information"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 30px;'>
        <h3 style='color: #ff4b4b;'>🔒 Credit Card Fraud Detection System</h3>
        <p style='font-size: 1.1rem;'><strong>Powered by XGBoost Machine Learning</strong></p>
        <p style='font-size: 0.95rem;'>Accuracy: 99.78% | Precision: 42.79% | Recall: 87.76% | F1-Score: 57.53%</p>
        <p style='font-size: 0.9rem; margin-top: 20px;'>Developed by <strong>Rajalekshmi Reji</strong></p>
        <p style='font-size: 0.85rem;'>ML Internship Level 3 - Challenge 2 | Certify Technology</p>
        <p style='font-size: 0.8rem; color: #888; margin-top: 15px;'>© 2026 | For educational purposes</p>
    </div>
""", unsafe_allow_html=True)