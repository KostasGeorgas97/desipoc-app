"""
DESIPOC Mental Health Risk Assessment - Streamlit Version
LSMU Database - BIPQ & EI Questionnaires
Standalone app that can be deployed to Streamlit Cloud
"""

import streamlit as st # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LSMU Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #764ba2;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .risk-moderate {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None

# Generate synthetic training data and train model
@st.cache_resource
def train_model():
    """Train Random Forest model on synthetic LSMU-like data"""
    np.random.seed(42)
    
    # Generate synthetic data matching LSMU distributions
    n_samples = 1000
    data = []
    
    for _ in range(n_samples):
        # Demographics
        age = np.random.normal(55, 12)
        gender = np.random.choice([1, 2], p=[0.7, 0.3])
        education = np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
        marital_status = np.random.choice([1, 2, 3, 4], p=[0.5, 0.2, 0.2, 0.1])
        residence = np.random.choice([1, 2], p=[0.3, 0.7])
        
        # Clinical
        cancer_type = np.random.choice(range(1, 18))
        cancer_stage = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
        time_since_diagnosis = np.random.exponential(12)
        first_treatment = np.random.choice([0, 1], p=[0.4, 0.6])
        chemotherapy = np.random.choice([0, 2], p=[0.3, 0.7])
        surgery = np.random.choice([0, 3], p=[0.2, 0.8])
        
        # BIPQ and EI (with correlation)
        bipq_sum = np.random.normal(45, 12)
        bipq_sum = np.clip(bipq_sum, 0, 80)
        
        ei_sum = 125 - 0.5 * bipq_sum + np.random.normal(0, 15)
        ei_sum = np.clip(ei_sum, 33, 165)
        
        # Create risk category
        if bipq_sum >= 50 and ei_sum < 111:
            risk = 2  # High
        elif bipq_sum < 42 and ei_sum > 137:
            risk = 0  # Low
        else:
            risk = 1  # Moderate
        
        data.append([age, gender, education, marital_status, residence,
                    cancer_type, cancer_stage, time_since_diagnosis,
                    first_treatment, chemotherapy, surgery, bipq_sum, ei_sum, risk])
    
    df = pd.DataFrame(data, columns=[
        'age', 'gender', 'education', 'marital_status', 'residence',
        'cancer_type', 'cancer_stage', 'time_since_diagnosis',
        'first_treatment', 'chemotherapy', 'surgery', 'bipq_sum', 'ei_sum', 'risk'
    ])
    
    # Train model
    feature_names = ['age', 'gender', 'education', 'marital_status', 'residence',
                    'cancer_type', 'cancer_stage', 'time_since_diagnosis',
                    'first_treatment', 'chemotherapy', 'surgery', 'bipq_sum', 'ei_sum']
    
    X = df[feature_names]
    y = df['risk']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, feature_names

# Load model
if st.session_state.model is None:
    with st.spinner('Training model...'):
        model, scaler, feature_names = train_model()
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names

# Title
st.markdown('<div class="main-header">🧠 LSMU Mental Health Assessment</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Cancer Patient Psychological Risk Assessment Tool</p>', unsafe_allow_html=True)

# Progress bar
progress = st.session_state.step / 7
st.progress(progress)
st.markdown(f"**Step {st.session_state.step + 1} of 7**")

# Navigation functions
def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def reset():
    st.session_state.step = 0
    st.session_state.data = {}

# Step 0: Demographics
if st.session_state.step == 0:
    st.markdown('<div class="sub-header">📊 Demographics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, key="age")
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male", key="gender")
        education = st.selectbox("Education Level", 
                                options=[1, 2, 3],
                                format_func=lambda x: ["Incomplete Secondary", "Secondary", "College/University"][x-1],
                                key="education")
    
    with col2:
        marital_status = st.selectbox("Marital Status",
                                     options=[1, 2, 3, 4],
                                     format_func=lambda x: ["Married", "Unmarried", "Divorced", "Widowed"][x-1],
                                     key="marital_status")
        residence = st.selectbox("Residence", 
                                options=[1, 2],
                                format_func=lambda x: "Rural" if x == 1 else "Urban",
                                key="residence")
    
    st.session_state.data.update({
        'age': age, 'gender': gender, 'education': education,
        'marital_status': marital_status, 'residence': residence
    })
    
    if st.button("Next →", type="primary"):
        next_step()
        st.rerun()

# Step 1: Clinical Information
elif st.session_state.step == 1:
    st.markdown('<div class="sub-header">🏥 Clinical Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cancer_types = ["Breast", "Lung", "Prostate", "Ovary", "Cervix", "Thyroid", 
                       "Colorectal", "Blood", "Liver", "Skin", "Stomach", "Brain",
                       "Bone Marrow", "Pelvis", "Throat", "Kidney", "Bile Duct"]
        cancer_type = st.selectbox("Cancer Type", options=range(1, 18),
                                  format_func=lambda x: cancer_types[x-1], key="cancer_type")
        
        cancer_stage = st.selectbox("Cancer Stage", options=[1, 2, 3, 4], key="cancer_stage")
        
        time_since_diagnosis = st.number_input("Time Since Diagnosis (months)", 
                                              min_value=0, max_value=240, value=6, key="time_since_diagnosis")
    
    with col2:
        first_treatment = st.selectbox("First Time in Treatment?", 
                                      options=[0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      key="first_treatment")
        
        chemotherapy = st.selectbox("Received Chemotherapy?",
                                   options=[0, 2],
                                   format_func=lambda x: "No" if x == 0 else "Yes",
                                   key="chemotherapy")
        
        surgery = st.selectbox("Received Surgery?",
                             options=[0, 3],
                             format_func=lambda x: "No" if x == 0 else "Yes",
                             key="surgery")
    
    st.session_state.data.update({
        'cancer_type': cancer_type, 'cancer_stage': cancer_stage,
        'time_since_diagnosis': time_since_diagnosis, 'first_treatment': first_treatment,
        'chemotherapy': chemotherapy, 'surgery': surgery
    })
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("← Previous"):
            prev_step()
            st.rerun()
    with col_b:
        if st.button("Next →", type="primary"):
            next_step()
            st.rerun()

# Step 2: BIPQ Questionnaire
elif st.session_state.step == 2:
    st.markdown('<div class="sub-header">📝 Brief Illness Perception Questionnaire (BIPQ)</div>', unsafe_allow_html=True)
    st.info("Please rate each question on a scale from 0-10")
    
    bipq_questions = [
        "How much does your illness affect your life?",
        "How long do you think your illness will continue?",
        "How much control do you feel you have over your illness?",
        "How much do you think your treatment can help your illness?",
        "How much do you experience symptoms from your illness?",
        "How concerned are you about your illness?",
        "How well do you feel you understand your illness?",
        "How much does your illness affect you emotionally?"
    ]
    
    bipq_responses = []
    for i, q in enumerate(bipq_questions):
        response = st.slider(f"{i+1}. {q}", 0, 10, 5, key=f"bipq_{i}")
        bipq_responses.append(response)
    
    # Calculate BIPQ sum (items 3 and 7 are reverse scored, item 4 not included in sum)
    bipq_sum = (bipq_responses[0] + bipq_responses[1] + (10 - bipq_responses[2]) +
                bipq_responses[4] + bipq_responses[5] + (10 - bipq_responses[6]) + 
                bipq_responses[7])
    
    st.session_state.data['bipq_sum'] = bipq_sum
    st.session_state.data['bipq_responses'] = bipq_responses
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("← Previous"):
            prev_step()
            st.rerun()
    with col_b:
        if st.button("Next →", type="primary"):
            next_step()
            st.rerun()

# Steps 3-5: EI Questionnaire (3 parts)
elif st.session_state.step in [3, 4, 5]:
    part = st.session_state.step - 2
    st.markdown(f'<div class="sub-header">📝 Emotional Intelligence - Part {part} of 3</div>', unsafe_allow_html=True)
    st.info("Rate how much you agree with each statement (1=Strongly Disagree, 5=Strongly Agree)")
    
    ei_questions = [
        # Part 1 (items 1-11)
        ["I know when to speak about my personal problems to others",
         "When I am faced with obstacles, I remember times I faced similar obstacles and overcame them",
         "I expect that I will do well on most things I try",
         "Other people find it easy to confide in me",
         "I find it hard to understand the non-verbal messages of other people",
         "Some of the major events of my life have led me to re-evaluate what is important and not important",
         "When my mood changes, I see new possibilities",
         "Emotions are one of the things that make my life worth living",
         "I am aware of my emotions as I experience them",
         "I expect good things to happen",
         "I like to share my emotions with others"],
        # Part 2 (items 12-22)
        ["When I experience a positive emotion, I know how to make it last",
         "I arrange events others enjoy",
         "I seek out activities that make me happy",
         "I am aware of the non-verbal messages I send to others",
         "I present myself in a way that makes a good impression on others",
         "When I am in a positive mood, solving problems is easy for me",
         "By looking at their facial expressions, I recognize the emotions people are experiencing",
         "I know why my emotions change",
         "When I am in a positive mood, I am able to come up with new ideas",
         "I have control over my emotions",
         "I easily recognize my emotions as I experience them"],
        # Part 3 (items 23-33)
        ["I motivate myself by imagining a good outcome to tasks I take on",
         "I compliment others when they have done something well",
         "I am aware of the non-verbal messages other people send",
         "When another person tells me about an important event in his or her life, I almost feel as though I experienced this event myself",
         "When I feel a change in emotions, I tend to come up with new ideas",
         "When I am faced with a challenge, I give up because I believe I will fail",
         "I know what other people are feeling just by looking at them",
         "I help other people feel better when they are down",
         "I use good moods to help myself keep trying in the face of obstacles",
         "I can tell how people are feeling by listening to the tone of their voice",
         "It is difficult for me to understand why people feel the way they do"]
    ]
    
    ei_responses = st.session_state.data.get('ei_responses', [])
    
    start_idx = (part - 1) * 11
    end_idx = start_idx + 11
    
    for i in range(start_idx, min(end_idx, 33)):
        q_idx = i - start_idx
        if part <= len(ei_questions) and q_idx < len(ei_questions[part-1]):
            response = st.slider(f"{i+1}. {ei_questions[part-1][q_idx]}", 1, 5, 3, key=f"ei_{i}")
            if len(ei_responses) <= i:
                ei_responses.append(response)
            else:
                ei_responses[i] = response
    
    st.session_state.data['ei_responses'] = ei_responses
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("← Previous"):
            prev_step()
            st.rerun()
    with col_b:
        if st.button("Next →", type="primary"):
            next_step()
            st.rerun()

# Step 6: Results
elif st.session_state.step == 6:
    st.markdown('<div class="sub-header">📊 Assessment Results</div>', unsafe_allow_html=True)
    
    # Calculate EI sum (items 5, 28, 33 are reverse scored)
    ei_responses = st.session_state.data['ei_responses']
    ei_sum = sum(ei_responses)
    # Apply reverse scoring
    ei_sum = ei_sum - ei_responses[4] + (6 - ei_responses[4])  # Item 5
    ei_sum = ei_sum - ei_responses[27] + (6 - ei_responses[27])  # Item 28
    ei_sum = ei_sum - ei_responses[32] + (6 - ei_responses[32])  # Item 33
    
    st.session_state.data['ei_sum'] = ei_sum
    
    # Prepare features for prediction
    feature_values = [
        st.session_state.data['age'],
        st.session_state.data['gender'],
        st.session_state.data['education'],
        st.session_state.data['marital_status'],
        st.session_state.data['residence'],
        st.session_state.data['cancer_type'],
        st.session_state.data['cancer_stage'],
        st.session_state.data['time_since_diagnosis'],
        st.session_state.data['first_treatment'],
        st.session_state.data['chemotherapy'],
        st.session_state.data['surgery'],
        st.session_state.data['bipq_sum'],
        ei_sum
    ]
    
    # Make prediction
    X = np.array([feature_values])
    X_scaled = st.session_state.scaler.transform(X)
    prediction = st.session_state.model.predict(X_scaled)[0]
    proba = st.session_state.model.predict_proba(X_scaled)[0]
    
    # Determine categories
    bipq_sum = st.session_state.data['bipq_sum']
    if bipq_sum < 42:
        bipq_cat = "Low Threat"
    elif bipq_sum < 50:
        bipq_cat = "Moderate Threat"
    else:
        bipq_cat = "High Threat"
    
    if ei_sum < 111:
        ei_cat = "Low EI"
    elif ei_sum <= 137:
        ei_cat = "Average EI"
    else:
        ei_cat = "High EI"
    
    risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]
    risk_colors = ["low", "moderate", "high"]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("BIPQ Score", f"{bipq_sum:.0f}/80")
        st.caption(bipq_cat)
    
    with col2:
        st.metric("EI Score", f"{ei_sum:.0f}/165")
        st.caption(ei_cat)
    
    with col3:
        st.metric("Risk Category", risk_labels[prediction])
        st.caption(f"Confidence: {proba[prediction]*100:.1f}%")
    
    # Risk assessment box
    st.markdown(f'<div class="risk-{risk_colors[prediction]}">', unsafe_allow_html=True)
    st.markdown(f"### {risk_labels[prediction]}")
    st.markdown(f"**Combined Profile:** {bipq_cat} + {ei_cat}")
    
    if prediction == 2:
        st.markdown("""
        **Recommendation:** High psychological risk detected. Consider referral to mental health services.
        - Cognitive behavioral therapy for illness perceptions
        - Emotional intelligence skills training
        - Regular psychological monitoring
        """)
    elif prediction == 1:
        st.markdown("""
        **Recommendation:** Moderate psychological risk. Supportive interventions recommended.
        - Psychoeducation about coping strategies
        - Support group participation
        - Monitor for changes
        """)
    else:
        st.markdown("""
        **Recommendation:** Low psychological risk. Continue regular monitoring.
        - Maintain current coping strategies
        - Routine psychological check-ins
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### Top Contributing Factors")
    importance = st.session_state.model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': st.session_state.feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(5)
    
    st.bar_chart(feature_imp.set_index('Feature'))
    
    # Demographics summary
    with st.expander("View Demographics Summary"):
        st.write(f"**Age:** {st.session_state.data['age']} years")
        st.write(f"**Gender:** {'Female' if st.session_state.data['gender'] == 1 else 'Male'}")
        st.write(f"**Education:** {['Incomplete Secondary', 'Secondary', 'College/University'][st.session_state.data['education']-1]}")
        st.write(f"**Cancer Stage:** {st.session_state.data['cancer_stage']}")
        st.write(f"**Time Since Diagnosis:** {st.session_state.data['time_since_diagnosis']} months")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("← Previous"):
            prev_step()
            st.rerun()
    with col_b:
        if st.button("Start New Assessment", type="primary"):
            reset()
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### About This Tool")
    st.info("""
    This assessment tool evaluates psychological risk in cancer patients using:
    - **BIPQ**: Brief Illness Perception Questionnaire
    - **EI**: Emotional Intelligence Scale
    
    Based on LSMU database research with validated cutoffs.
    """)
    
    st.markdown("### Risk Categories")
    st.markdown("""
    **BIPQ Categories:**
    - Low: <42
    - Moderate: 42-49
    - High: ≥50
    
    **EI Categories:**
    - Low: <111
    - Average: 111-137
    - High: >137
    """)
    
    if st.button("Reset Assessment"):
        reset()
        st.rerun()