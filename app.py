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
        st.info("""
        **Troubleshooting Steps:**
        1. Make sure 'fraud_detection_model.pkl' is in the same folder as app.py
        2. Make sure 'scaler.pkl' is in the same folder as app.py
        3. Check that the files are not corrupted
        4. Try re-downloading the files from Google Colab
        """)
        st.stop()

# Load models
model, scaler = load_models()

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
    
    st.warning("""
    ⚠️ **Important Note:**
    - **Precision: 42.79%** means ~58% of fraud alerts are false positives
    - Best used with manual review for 30-70% probability range
    - Threshold adjustment recommended for production
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "99.78%", delta="High", delta_color="normal")
        st.metric("Recall", "87.76%", delta="Good", delta_color="normal")
        st.metric("ROC-AUC", "98.36%", delta="Excellent", delta_color="normal")
    with col2:
        st.metric("Precision", "42.79%", delta="Fair", delta_color="inverse")
        st.metric("F1-Score", "57.53%", delta="Good", delta_color="normal")
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    st.write("• **Total Transactions:** 284,807")
    st.write("• **Legitimate:** 284,315 (99.83%)")
    st.write("• **Fraudulent:** 492 (0.17%)")
    st.write("• **Features Used:** 30")
    st.write("• **Imbalance Ratio:** 1:578")
    
    st.markdown("---")
    st.markdown("### 🎯 Model Capabilities")
    st.success("✓ Real-time fraud detection")
    st.success("✓ Probability scoring (0-100%)")
    st.success("✓ Risk level classification")
    st.success("✓ High recall (87.76%)")
    
    st.markdown("---")
    st.markdown("### 📋 Risk Level Guide")
    st.markdown("""
    - **Class 1 Prediction**: 🔴 FRAUD - Decline
    - **Class 0 + 30-70%**: 🟡 Review needed
    - **Class 0 + <30%**: 🟢 Approve
    """)

# Main content
st.markdown("## 🔍 Single Transaction Analysis")
st.markdown("Enter transaction details below to check for potential fraud.")

# Input section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 💳 Transaction Details")
    time_seconds = st.number_input(
        "⏱️ Time (seconds since first transaction)", 
        min_value=0.0, 
        max_value=200000.0,
        value=50000.0, 
        step=1000.0,
        help="Time elapsed since the first transaction in the dataset"
    )
    amount = st.number_input(
        "💵 Transaction Amount ($)", 
        min_value=0.0, 
        max_value=30000.0,
        value=100.0, 
        step=10.0,
        help="Transaction amount in US dollars"
    )

with col2:
    st.markdown("### 🔢 PCA Features")
    st.info("📌 Principal Component Analysis features (V1-V28). These are anonymized features derived from the original transaction data.")
    
    preset = st.selectbox(
        "🎯 Quick Test Presets",
        ["Custom Values", "Known Legitimate Transaction", "Known Fraudulent Transaction"],
        help="Select a preset to quickly test the model"
    )

# PCA Features Input
st.markdown("---")
st.markdown("### 🔢 Enter V1-V28 Feature Values")

features = []

if preset == "Known Fraudulent Transaction":
    fraud_values = [
        -2.3122265423263,  # V1
        1.95199201064158,  # V2
        -1.60985073229769,  # V3
        3.9979055875468,  # V4
        -0.522187864667764,  # V5
        -1.42654531920595,  # V6
        -2.53738730624579,  # V7
        1.39165724829804,  # V8
        -2.77008927719433,  # V9
        -2.77227214465915,  # V10
        3.20203320709635,  # V11
        -2.89990738849473,  # V12
        -0.595221881324605,  # V13
        -4.28925378244217,  # V14
        0.389724120274487,  # V15
        -1.14074717980657,  # V16
        -2.83005567450437,  # V17
        -0.0168224681808257,  # V18
        0.416955705037907,  # V19
        0.126910559061474,  # V20
        0.517232370861764,  # V21
        -0.0350493686052974,  # V22
        -0.465211076182388,  # V23
        0.320198198514526,  # V24
        0.0445191674731724,  # V25
        0.177839798284401,  # V26
        0.261145002567677,  # V27
        -0.143275874698919,  # V28
    ]
    features = fraud_values
    time_seconds = 406.0
    amount = 0.0
    
    st.warning("⚠️ Using known fraudulent transaction values")
    
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.text_input(f"V{i+1}", value=f"{fraud_values[i]:.6f}", disabled=True, key=f"fraud_v{i+1}")

elif preset == "Known Legitimate Transaction":
    features = [0.0] * 28
    time_seconds = 50000.0
    amount = 100.0
    
    st.success("✅ Using known legitimate transaction values")
    
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
                key=f"custom_v{i+1}",
                help=f"Principal Component {i+1}"
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
            # Prepare input data
            input_data = np.array([[time_seconds] + features + [amount]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            # CORRECTED INTERPRETATION
            prediction_class = int(prediction[0])  # 0 = Legitimate, 1 = Fraud
            fraud_probability = float(probability[0][1] * 100)
            legit_probability = float(probability[0][0] * 100)
            confidence = float(max(probability[0]) * 100)
            
            # Determine classification based on ACTUAL prediction
            if prediction_class == 1:
                # Model says FRAUD
                classification = "FRAUDULENT"
                risk_level = "🔴 HIGH RISK"
                risk_color = "red"
                recommendation = "DECLINE TRANSACTION"
            elif fraud_probability >= 30:
                # Model says legitimate but probability is medium
                classification = "LEGITIMATE (SUSPICIOUS)"
                risk_level = "🟠 MEDIUM RISK"
                risk_color = "orange"
                recommendation = "MANUAL REVIEW REQUIRED"
            else:
                # Model says legitimate and low fraud probability
                classification = "LEGITIMATE"
                risk_level = "🟢 LOW RISK"
                risk_color = "green"
                recommendation = "APPROVE TRANSACTION"
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if prediction_class == 1:
                    st.metric("🚨 Classification", "FRAUD", delta="Critical")
                elif fraud_probability >= 30:
                    st.metric("⚠️ Classification", "SUSPICIOUS", delta="Review")
                else:
                    st.metric("✅ Classification", "LEGITIMATE", delta="Safe")
            
            with col2:
                st.metric("🎲 Fraud Probability", f"{fraud_probability:.2f}%")
            
            with col3:
                st.metric("✓ Model Confidence", f"{confidence:.2f}%")
            
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
                # MODEL PREDICTED FRAUD
                st.error(f"""
                ### 🚨 HIGH RISK - FRAUDULENT TRANSACTION DETECTED
                
                **Risk Assessment:**
                - **Classification:** 🔴 FRAUDULENT (Model Prediction: Class {prediction_class})
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Legitimate Probability:** {legit_probability:.2f}%
                - **Model Confidence:** {confidence:.2f}%
                - **Risk Level:** {risk_level}
                - **Recommendation:** ⛔ **{recommendation}**
                
                **Transaction Details:**
                - 💵 **Amount:** ${amount:,.2f}
                - ⏱️ **Time:** {time_seconds:,.0f} seconds
                
                **Immediate Actions Required:**
                1. 🚫 **DECLINE** this transaction immediately
                2. 📞 **Alert** the cardholder via registered phone/email
                3. 🔍 **Review** recent transaction history for patterns
                4. 🔒 **Consider** temporary card suspension
                5. 📊 **Log** incident for fraud analytics
                
                ⚠️ **Note:** With 42.79% precision, there's a 57% chance this is a false positive. 
                Always verify with cardholder before permanent account actions.
                """)
                
            elif fraud_probability >= 30:
                # MODEL PREDICTED LEGITIMATE BUT PROBABILITY IS MEDIUM
                st.warning(f"""
                ### ⚠️ MEDIUM RISK - MANUAL REVIEW REQUIRED
                
                **Risk Assessment:**
                - **Classification:** 🟡 {classification} (Model Prediction: Class {prediction_class})
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Legitimate Probability:** {legit_probability:.2f}%
                - **Model Confidence:** {confidence:.2f}%
                - **Risk Level:** {risk_level}
                - **Recommendation:** 🔍 **{recommendation}**
                
                **Transaction Details:**
                - 💵 **Amount:** ${amount:,.2f}
                - ⏱️ **Time:** {time_seconds:,.0f} seconds
                
                **Recommended Actions:**
                1. 📞 **Contact** cardholder for verification before processing
                2. ✅ **Approve** if cardholder confirms transaction
                3. 🚫 **Decline** if unable to reach cardholder or suspicious response
                4. 📝 **Document** verification attempt and outcome
                5. 👀 **Monitor** account for next 24-48 hours
                
                💡 **Best Practice:** Model classified as legitimate (Class 0) but fraud probability 
                is in the uncertain range. Human judgment required.
                """)
                
            else:
                # MODEL PREDICTED LEGITIMATE AND LOW PROBABILITY
                st.success(f"""
                ### ✅ LOW RISK - LEGITIMATE TRANSACTION
                
                **Risk Assessment:**
                - **Classification:** 🟢 {classification} (Model Prediction: Class {prediction_class})
                - **Fraud Probability:** {fraud_probability:.2f}%
                - **Legitimate Probability:** {legit_probability:.2f}%
                - **Model Confidence:** {confidence:.2f}%
                - **Risk Level:** {risk_level}
                - **Recommendation:** ✅ **{recommendation}**
                
                **Transaction Details:**
                - 💵 **Amount:** ${amount:,.2f}
                - ⏱️ **Time:** {time_seconds:,.0f} seconds
                
                **Action:**
                - ✓ Process transaction normally
                - ✓ No additional verification required
                - ✓ Standard security protocols apply
                
                **Note:** Continue monitoring for unusual patterns. Even legitimate 
                transactions should be part of ongoing fraud prevention analysis.
                """)
            
            # Technical details
            with st.expander("🔧 Technical Details & Model Performance"):
                st.markdown("### Model Prediction Details")
                st.write(f"**Raw Prediction:** {prediction_class} (0=Legitimate, 1=Fraud)")
                st.write(f"**Probability Vector:** [Legit: {legit_probability:.2f}%, Fraud: {fraud_probability:.2f}%]")
                st.write(f"**Input Shape:** {input_scaled.shape}")
                st.write(f"**Features Used:** 30 (Time + 28 PCA components + Amount)")
                
                st.markdown("### Model Performance Characteristics")
                st.write("""
                - **Recall: 87.76%** - Catches ~88% of actual fraud
                - **Precision: 42.79%** - ~43% of fraud alerts are true fraud
                - **False Positive Rate: ~58%** - Many legitimate transactions flagged
                
                **Interpretation:**
                - Excellent at catching fraud (high recall)
                - Generates many false alarms (low precision)
                - Best used with manual review layer
                - Consider threshold adjustment for production
                """)
                
                st.markdown("### Input Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['Time', 'Amount'] + [f'V{i+1}' for i in range(28)],
                    'Value': [time_seconds, amount] + features
                })
                st.dataframe(summary_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"""
            ### ❌ Error During Prediction
            
            **Error Message:** {str(e)}
            
            **Possible Causes:**
            - Invalid input values
            - Model compatibility issues
            - Feature scaling problems
            
            **Troubleshooting:**
            1. Check that all V1-V28 values are numeric
            2. Ensure Time and Amount are positive numbers
            3. Try using one of the preset test cases
            4. Verify model and scaler files are correctly loaded
            
            Please try again or contact support.
            """)

# Additional information section
st.markdown("---")
st.markdown("## 📚 How to Use This System")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Input Method
    - Enter transaction time (seconds)
    - Enter transaction amount ($)
    - Enter V1-V28 PCA features
    - OR use quick test presets
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Analysis
    - Click "Analyze Transaction"
    - Model processes input
    - Prediction generated instantly
    - Results displayed with details
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Action
    - Review fraud probability
    - Check risk level
    - Follow recommendations
    - Take appropriate action
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🔒 Credit Card Fraud Detection System</h3>
        <p><strong>Powered by XGBoost Machine Learning</strong></p>
        <p>Model Accuracy: 99.78% | Recall: 87.76% | ROC-AUC: 98.36%</p>
        <p>Developed by <strong>Rajalekshmi Reji</strong></p>
        <p>Machine Learning Internship Level 3 - Challenge 2</p>
        <p>Certify Technology | January 2026</p>
        <hr style='width: 50%; margin: 20px auto;'>
        <p style='font-size: 0.9rem;'>
            This system is designed for educational and demonstration purposes.<br>
            For production deployment, additional security measures and compliance checks are required.
        </p>
    </div>
""", unsafe_allow_html=True)