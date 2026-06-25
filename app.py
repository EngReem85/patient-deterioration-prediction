import gradio as gr
import numpy as np
import pandas as pd
import pickle
import warnings
import plotly.graph_objects as go
from datetime import datetime
warnings.filterwarnings('ignore')
from huggingface_hub import hf_hub_download
import os

def load_model():
    try:
        
        local_path = 'model_artifacts(1).pkl'
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                artifacts = pickle.load(f)
            print("✅ Model loaded successfully from local file")
            return artifacts.get('model'), artifacts.get('scaler')
        
        
        print("📥 Downloading model from Hugging Face Hub...")
        model_path = hf_hub_download(
            repo_id="EngReem85/patient-deterioration-predictor",  
            filename="model_artifacts(1).pkl"
        )
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        print("✅ Model loaded successfully from Hub")
        return artifacts.get('model'), artifacts.get('scaler')
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

model, scaler = load_model()

def analyze_trends(historical_df):
    """Analyze trends over time"""
    if historical_df is None or len(historical_df) < 3:
        return {
            'risk_level': 'INSUFFICIENT_DATA',
            'risk_points': 0,
            'trend_reasons': ['Not enough data for trend analysis']
        }
    
    try:
        win = historical_df.copy()
        cols = ["heart_rate", "respiratory_rate", "spo2_pct", 
                "systolic_bp", "diastolic_bp", "mobility_score"]
        
        features = {}
        for c in cols:
            if c in win.columns:
                y = pd.to_numeric(win[c], errors='coerce')
                if len(y.dropna()) >= 3:
                    x = np.arange(len(y))
                    coeffs = np.polyfit(x[~np.isnan(y)], y.dropna(), 1)
                    features[f"{c}_slope"] = float(coeffs[0])
                    features[f"{c}_delta"] = float(y.iloc[-1] - y.iloc[0])
        
        
        risk_points = 0
        reasons = []
        
        if features.get('heart_rate_slope', 0) > 1.0:
            risk_points += 2
            reasons.append(f"Heart rate trending up: {features['heart_rate_slope']:.2f}/hour")
        
        if features.get('respiratory_rate_slope', 0) > 0.5:
            risk_points += 1
            reasons.append(f"Respiratory rate trending up: {features['respiratory_rate_slope']:.2f}/hour")
        
        if features.get('spo2_pct_slope', 0) < -0.3:
            risk_points += 1
            reasons.append(f"Oxygen saturation trending down: {features['spo2_pct_slope']:.2f}%/hour")
        
        if abs(features.get('systolic_bp_slope', 10)) < 0.8:
            risk_points += 1
            reasons.append("Blood pressure not improving")
        
        if features.get('mobility_score_delta', 0) < -1:
            risk_points += 1
            reasons.append("Mobility declining")
        
        if risk_points >= 4:
            risk_level = "HIGH"
        elif risk_points >= 2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_points': risk_points,
            'trend_reasons': reasons,
            'features': features
        }
        
    except Exception as e:
        return {
            'risk_level': 'ERROR',
            'risk_points': 0,
            'trend_reasons': [f"Analysis error: {str(e)}"]
        }

def predict_deterioration(
    heart_rate, respiratory_rate, spo2_pct, temperature_c,
    systolic_bp, diastolic_bp, mobility_score, nurse_alert,
    wbc_count, lactate, creatinine, crp_level, hemoglobin,
    sepsis_risk_score, age, baseline_risk_score,
    
    hr_history_1, hr_history_2, hr_history_3,
    rr_history_1, rr_history_2, rr_history_3,
    spo2_history_1, spo2_history_2, spo2_history_3,
    sbp_history_1, sbp_history_2, sbp_history_3,
    dbp_history_1, dbp_history_2, dbp_history_3,
    mob_history_1, mob_history_2, mob_history_3
):
    """
    Combined ML and trend analysis prediction
    """
   
    ml_risk = None
    ml_recommendation = "ML model not available"
    
    if model and scaler:
        try:
            
            input_values = [
                float(heart_rate), float(respiratory_rate), float(spo2_pct), float(temperature_c),
                float(systolic_bp), float(diastolic_bp), 0, 0,  # oxygen_device, oxygen_flow
                float(mobility_score), float(nurse_alert), float(wbc_count), float(lactate),
                float(creatinine), float(crp_level), float(hemoglobin), float(sepsis_risk_score),
                float(age), 0, 2.0, 0, float(baseline_risk_score)  # gender, comorbidity, admission_type
            ]
            
            X = np.array(input_values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            ml_prob = model.predict_proba(X_scaled)[0][1]
            ml_risk = ml_prob * 100
            
            if ml_risk >= 70:
                ml_recommendation = "🔴 HIGH ML RISK: Immediate attention required"
            elif ml_risk >= 40:
                ml_recommendation = "🟡 MODERATE ML RISK: Close monitoring needed"
            else:
                ml_recommendation = "🟢 LOW ML RISK: Routine monitoring"
                
        except Exception as e:
            print(f"ML error: {e}")
            ml_risk = 30.0  # Default value
            ml_recommendation = "⚠️ ML prediction using default value"
    

    
    historical_data = pd.DataFrame({
        'hour': [1, 2, 3, 4, 5, 6],
        'heart_rate': [float(hr_history_1), float(hr_history_2), float(hr_history_3), 
                      float(heart_rate)-3, float(heart_rate)-1, float(heart_rate)],
        'respiratory_rate': [float(rr_history_1), float(rr_history_2), float(rr_history_3),
                           float(respiratory_rate)-2, float(respiratory_rate)-1, float(respiratory_rate)],
        'spo2_pct': [float(spo2_history_1), float(spo2_history_2), float(spo2_history_3),
                    float(spo2_pct)+2, float(spo2_pct)+1, float(spo2_pct)],
        'systolic_bp': [float(sbp_history_1), float(sbp_history_2), float(sbp_history_3),
                       float(systolic_bp)+5, float(systolic_bp)+2, float(systolic_bp)],
        'diastolic_bp': [float(dbp_history_1), float(dbp_history_2), float(dbp_history_3),
                        float(diastolic_bp)+4, float(diastolic_bp)+2, float(diastolic_bp)],
        'mobility_score': [float(mob_history_1), float(mob_history_2), float(mob_history_3),
                          float(mobility_score)+1, float(mobility_score), float(mobility_score)]
    })
    
    trend_analysis = analyze_trends(historical_data)
    
    
    final_risk = "LOW"
    reasons = []
    
    
    if (ml_risk and ml_risk >= 70) or (trend_analysis['risk_level'] == 'HIGH'):
        final_risk = "🔴 HIGH RISK"
        if ml_risk and ml_risk >= 70:
            reasons.append(f"ML predicts high risk ({ml_risk:.1f}%)")
        if trend_analysis['risk_level'] == 'HIGH':
            reasons.append(f"Trend analysis shows high risk ({trend_analysis['risk_points']} points)")
    
    elif (ml_risk and ml_risk >= 40) or (trend_analysis['risk_level'] == 'MODERATE'):
        final_risk = "🟡 MODERATE RISK"
        reasons.append("Increased monitoring recommended")
    
    else:
        final_risk = "🟢 LOW RISK"
        reasons.append("Routine monitoring advised")
    
    
    if trend_analysis['trend_reasons']:
        reasons.extend(trend_analysis['trend_reasons'])
    
    
    risk_value = ml_risk if ml_risk else (trend_analysis['risk_points'] * 15)
    fig = create_risk_gauge(risk_value, final_risk)
    
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "final_risk": final_risk,
        "ml_score": f"{ml_risk:.1f}%" if ml_risk else "N/A",
        "trend_score": f"{trend_analysis['risk_points']} points",
        "trend_level": trend_analysis['risk_level'],
        "reasons": reasons[:3],  # Show top 3 reasons
        "recommendation": ml_recommendation if ml_risk else "Use clinical judgment"
    }
    
    return final_risk, f"{ml_risk:.1f}%" if ml_risk else "N/A", \
           f"{trend_analysis['risk_points']} points", \
           "\n".join(reasons[:3]), results, fig


def create_risk_gauge(value, risk_level):
    """Create risk gauge visualization"""
    
    if "HIGH" in risk_level:
        color = "red"
        title_text = "HIGH RISK"
    elif "MODERATE" in risk_level:
        color = "orange"
        title_text = "MODERATE RISK"
    else:
        color = "green"
        title_text = "LOW RISK"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(value, 100),
        title={'text': title_text, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ]
        }
    ))
    
    fig.update_layout(height=300)
    return fig


with gr.Blocks(theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🏥 Patient Deterioration Prediction")
    gr.Markdown("Combines real-time assessment with trend analysis")
    
    with gr.Tabs():
        
        with gr.Tab("📋 Current Vital Signs"):
            with gr.Row():
                with gr.Column():
                    heart_rate = gr.Slider(40, 180, value=75, label="Heart Rate (bpm)", interactive=True)
                    respiratory_rate = gr.Slider(8, 40, value=16, label="Respiratory Rate", interactive=True)
                    spo2_pct = gr.Slider(70, 100, value=96, label="Oxygen Saturation (%)", interactive=True)
                    temperature_c = gr.Slider(35.0, 42.0, value=37.0, label="Temperature (°C)", interactive=True)
                
                with gr.Column():
                    systolic_bp = gr.Slider(70, 200, value=120, label="Systolic BP (mmHg)", interactive=True)
                    diastolic_bp = gr.Slider(40, 120, value=80, label="Diastolic BP (mmHg)", interactive=True)
                    mobility_score = gr.Slider(1, 4, value=3, step=1, label="Mobility Score", interactive=True)
                    nurse_alert = gr.Slider(0, 1, value=0, step=1, label="Nurse Alert", interactive=True)
            
            with gr.Row():
                with gr.Column():
                    wbc_count = gr.Slider(0.1, 50.0, value=7.5, label="White Blood Cells", interactive=True)
                    lactate = gr.Slider(0.1, 10.0, value=1.2, label="Lactate", interactive=True)
                    creatinine = gr.Slider(0.1, 10.0, value=1.0, label="Creatinine", interactive=True)
                
                with gr.Column():
                    crp_level = gr.Slider(0.1, 200.0, value=10.0, label="CRP", interactive=True)
                    hemoglobin = gr.Slider(5.0, 20.0, value=13.5, label="Hemoglobin", interactive=True)
                    sepsis_risk_score = gr.Slider(0.0, 1.0, value=0.3, label="Sepsis Risk", interactive=True)
            
            with gr.Row():
                age = gr.Slider(18, 100, value=45, label="Age", interactive=True)
                baseline_risk_score = gr.Slider(0.0, 1.0, value=0.2, label="Baseline Risk", interactive=True)
        
        
        with gr.Tab("📈 Trend Analysis Input"):
            gr.Markdown("### Enter values from previous hours for trend analysis")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 3 Hours Ago")
                    hr_history_1 = gr.Slider(40, 180, value=70, label="Heart Rate", interactive=True)
                    rr_history_1 = gr.Slider(8, 40, value=15, label="Respiratory Rate", interactive=True)
                    spo2_history_1 = gr.Slider(70, 100, value=98, label="Oxygen Saturation", interactive=True)
                    sbp_history_1 = gr.Slider(70, 200, value=125, label="Systolic BP", interactive=True)
                    dbp_history_1 = gr.Slider(40, 120, value=85, label="Diastolic BP", interactive=True)
                    mob_history_1 = gr.Slider(1, 4, value=4, label="Mobility Score", interactive=True)
                
                with gr.Column():
                    gr.Markdown("#### 2 Hours Ago")
                    hr_history_2 = gr.Slider(40, 180, value=72, label="Heart Rate", interactive=True)
                    rr_history_2 = gr.Slider(8, 40, value=15, label="Respiratory Rate", interactive=True)
                    spo2_history_2 = gr.Slider(70, 100, value=97, label="Oxygen Saturation", interactive=True)
                    sbp_history_2 = gr.Slider(70, 200, value=122, label="Systolic BP", interactive=True)
                    dbp_history_2 = gr.Slider(40, 120, value=83, label="Diastolic BP", interactive=True)
                    mob_history_2 = gr.Slider(1, 4, value=4, label="Mobility Score", interactive=True)
                
                with gr.Column():
                    gr.Markdown("#### 1 Hour Ago")
                    hr_history_3 = gr.Slider(40, 180, value=74, label="Heart Rate", interactive=True)
                    rr_history_3 = gr.Slider(8, 40, value=16, label="Respiratory Rate", interactive=True)
                    spo2_history_3 = gr.Slider(70, 100, value=96, label="Oxygen Saturation", interactive=True)
                    sbp_history_3 = gr.Slider(70, 200, value=120, label="Systolic BP", interactive=True)
                    dbp_history_3 = gr.Slider(40, 120, value=82, label="Diastolic BP", interactive=True)
                    mob_history_3 = gr.Slider(1, 4, value=3, label="Mobility Score", interactive=True)
    
    
    predict_btn = gr.Button("🚀 Predict Deterioration Risk", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    
    gr.Markdown("## 📊 Prediction Results")
    
    with gr.Row():
        with gr.Column():
            final_risk_output = gr.Textbox(label="Final Risk Level")
            ml_score_output = gr.Textbox(label="ML Risk Score")
            trend_score_output = gr.Textbox(label="Trend Analysis Score")
        
        with gr.Column():
            gauge_output = gr.Plot(label="Risk Visualization")
    
    with gr.Row():
        reasons_output = gr.Textbox(label="Risk Reasons", lines=3)
        results_json = gr.JSON(label="Detailed Results")
    
    # Connect Button to Function
    predict_btn.click(
        fn=predict_deterioration,
        inputs=[
            heart_rate, respiratory_rate, spo2_pct, temperature_c,
            systolic_bp, diastolic_bp, mobility_score, nurse_alert,
            wbc_count, lactate, creatinine, crp_level, hemoglobin,
            sepsis_risk_score, age, baseline_risk_score,
            hr_history_1, hr_history_2, hr_history_3,
            rr_history_1, rr_history_2, rr_history_3,
            spo2_history_1, spo2_history_2, spo2_history_3,
            sbp_history_1, sbp_history_2, sbp_history_3,
            dbp_history_1, dbp_history_2, dbp_history_3,
            mob_history_1, mob_history_2, mob_history_3
        ],
        outputs=[
            final_risk_output, ml_score_output, trend_score_output,
            reasons_output, results_json, gauge_output
        ]
    )
    
    
    clear_btn = gr.Button("🔄 Reset All Values", variant="secondary")
    
    def clear_all():
        
        defaults = []
        
        defaults.extend([75, 16, 96, 37.0, 120, 80, 3, 0, 7.5, 1.2, 1.0, 10.0, 13.5, 0.3, 45, 0.2])
        
        defaults.extend([70, 72, 74, 15, 15, 16, 98, 97, 96, 125, 122, 120, 85, 83, 82, 4, 4, 3])
        
        defaults.extend(["", "", "", "", {}, None])
        return defaults
    
    clear_btn.click(
        fn=clear_all,
        outputs=[
            heart_rate, respiratory_rate, spo2_pct, temperature_c,
            systolic_bp, diastolic_bp, mobility_score, nurse_alert,
            wbc_count, lactate, creatinine, crp_level, hemoglobin,
            sepsis_risk_score, age, baseline_risk_score,
            hr_history_1, hr_history_2, hr_history_3,
            rr_history_1, rr_history_2, rr_history_3,
            spo2_history_1, spo2_history_2, spo2_history_3,
            sbp_history_1, sbp_history_2, sbp_history_3,
            dbp_history_1, dbp_history_2, dbp_history_3,
            mob_history_1, mob_history_2, mob_history_3,
            final_risk_output, ml_score_output, trend_score_output,
            reasons_output, results_json, gauge_output
        ]
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)                                                              
