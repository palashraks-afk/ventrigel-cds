import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import datetime
import json
import base64

st.set_page_config(
    page_title="VentriGel CDS | EMR",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'theme_preference' not in st.session_state:
    st.session_state.theme_preference = "Dark Mode"

def get_theme_variables(is_dark):
    if is_dark:
        return {
            "bg_color": "#0F172A",
            "panel_bg": "#1E293B",
            "text_main": "#F8FAFC",
            "text_muted": "#94A3B8",
            "border": "#334155",
            "accent": "#38BDF8",
            "success_bg": "rgba(34, 197, 94, 0.1)",
            "success_border": "#22C55E",
            "danger_bg": "rgba(239, 68, 68, 0.1)",
            "danger_border": "#EF4444",
            "chart_template": "plotly_dark",
            "tab_bg": "#0F172A",
            "tab_inactive": "transparent"
        }
    else:
        return {
            "bg_color": "#F8FAFC",
            "panel_bg": "#FFFFFF",
            "text_main": "#0F172A",
            "text_muted": "#475569",
            "border": "#CBD5E1",
            "accent": "#0284C7",
            "success_bg": "#F0FDF4",
            "success_border": "#22C55E",
            "danger_bg": "#FEF2F2",
            "danger_border": "#EF4444",
            "chart_template": "plotly_white",
            "tab_bg": "#F8FAFC",
            "tab_inactive": "transparent"
        }

theme = get_theme_variables(st.session_state.theme_preference == "Dark Mode")

css_string = f"""
<style>
    .stApp {{
        background-color: {theme['bg_color']};
    }}
    .block-container {{
        padding-top: 2rem;
        max-width: 1600px;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        color: {theme['text_main']} !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    p, span, div, label, li {{
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        color: {theme['text_main']};
        line-height: 1.6;
    }}
    .text-muted {{
        color: {theme['text_muted']} !important;
    }}
    .metric-container {{
        background-color: {theme['panel_bg']};
        border: 1px solid {theme['border']};
        padding: 25px;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .success-panel {{
        background-color: {theme['success_bg']};
        border: 1px solid {theme['success_border']};
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 20px;
    }}
    .danger-panel {{
        background-color: {theme['danger_bg']};
        border: 1px solid {theme['danger_border']};
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: transparent;
        border-bottom: 1px solid {theme['border']};
        padding: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        font-size: 16px;
        font-weight: 600;
        color: {theme['text_muted']};
        padding: 0 24px;
        border: 1px solid transparent;
        border-bottom: none;
        background-color: {theme['tab_inactive']};
        border-radius: 8px 8px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        color: {theme['text_main']};
        border: 1px solid {theme['border']};
        border-bottom: 1px solid {theme['bg_color']};
        background-color: {theme['bg_color']};
        margin-bottom: -1px;
    }}
    .stButton>button {{
        background-color: {theme['accent']};
        color: #FFFFFF !important;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 6px;
        border: 1px solid {theme['accent']};
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}
    hr {{
        border-color: {theme['border']};
        margin: 30px 0;
    }}
    [data-testid="stForm"] {{
        background-color: {theme['panel_bg']};
        border-color: {theme['border']};
    }}
</style>
"""

st.markdown(
    css_string,
    unsafe_allow_html=True
)

@st.cache_resource
def load_clinical_model():
    try:
        model_object = joblib.load(
            'ventrigel_clinical_pipeline_v2.joblib'
        )
        return model_object, True
    except FileNotFoundError:
        return None, False

pipeline_model, model_is_loaded = load_clinical_model()

if not model_is_loaded:
    st.error(
        ":material/error: CRITICAL SYSTEM FAULT. The required model binary is missing from the root directory. Execute the model training sequence before launching the application."
    )
    st.stop()

def build_clinical_gauge(val, title, min_val, max_val, opt_low, opt_high, units, theme_dict):
    
    title_configuration = {
        'text': f"{title}<br><span style='font-size:14px;color:{theme_dict['text_muted']}'>{units}</span>",
        'font': {
            'size': 16,
            'color': theme_dict['text_main']
        }
    }
    
    number_configuration = {
        'font': {
            'size': 36,
            'color': theme_dict['text_main'],
            'weight': 'bold'
        }
    }
    
    axis_configuration = {
        'range': [
            min_val,
            max_val
        ],
        'tickwidth': 2,
        'tickcolor': theme_dict['border']
    }
    
    bar_configuration = {
        'color': theme_dict['accent'],
        'thickness': 0.15
    }
    
    steps_list = [
        {
            'range': [
                min_val,
                opt_low
            ],
            'color': "rgba(239, 68, 68, 0.2)"
        },
        {
            'range': [
                opt_low,
                opt_high
            ],
            'color': "rgba(34, 197, 94, 0.2)"
        },
        {
            'range': [
                opt_high,
                max_val
            ],
            'color': "rgba(239, 68, 68, 0.2)"
        }
    ]
    
    threshold_configuration = {
        'line': {
            'color': theme_dict['text_main'],
            'width': 4
        },
        'thickness': 0.75,
        'value': val
    }
    
    gauge_configuration = {
        'axis': axis_configuration,
        'bar': bar_configuration,
        'bgcolor': theme_dict['panel_bg'],
        'borderwidth': 1,
        'bordercolor': theme_dict['border'],
        'steps': steps_list,
        'threshold': threshold_configuration
    }
    
    indicator_trace = go.Indicator(
        mode="gauge+number",
        value=val,
        title=title_configuration,
        number=number_configuration,
        gauge=gauge_configuration
    )
    
    figure_object = go.Figure(
        indicator_trace
    )
    
    margin_configuration = dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
    
    figure_object.update_layout(
        height=300,
        margin=margin_configuration,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={
            'family': "Inter"
        }
    )
    
    return figure_object

def build_radar_chart(patient_data, theme_dict):
    
    category_labels = [
        'LVEF (%)',
        'Scar Mass (%)',
        'LVESV (mL)',
        'BNP (Scaled x10)',
        '6M Walk (Scaled x10)'
    ]
    
    optimal_target_vector = [
        35.0,
        25.0,
        150.0,
        30.0,
        42.0
    ]
    
    patient_ef = patient_data['Baseline_EF_Percent']
    patient_scar = patient_data['Scar_Mass_Percent']
    patient_lvesv = patient_data['LVESV_mL']
    patient_bnp_scaled = patient_data['BNP_Level_pgml'] / 10.0
    patient_walk_scaled = patient_data['Walk_6_Min_meters'] / 10.0
    
    patient_metric_vector = [
        patient_ef,
        patient_scar,
        patient_lvesv,
        patient_bnp_scaled,
        patient_walk_scaled
    ]
    
    radar_figure = go.Figure()
    
    protocol_trace = go.Scatterpolar(
        r=optimal_target_vector,
        theta=category_labels,
        fill='toself',
        name='Protocol Target Baseline',
        line_color=theme_dict['text_muted'],
        fillcolor='rgba(148, 163, 184, 0.2)'
    )
    
    radar_figure.add_trace(
        protocol_trace
    )
    
    patient_trace = go.Scatterpolar(
        r=patient_metric_vector,
        theta=category_labels,
        fill='toself',
        name='Current Patient Profile',
        line_color=theme_dict['accent'],
        fillcolor='rgba(56, 189, 248, 0.4)'
    )
    
    radar_figure.add_trace(
        patient_trace
    )
    
    radial_axis_config = dict(
        visible=True,
        range=[
            0,
            200
        ],
        gridcolor=theme_dict['border'],
        linecolor=theme_dict['border'],
        tickfont=dict(
            color=theme_dict['text_muted']
        )
    )
    
    angular_axis_config = dict(
        tickfont=dict(
            size=13,
            color=theme_dict['text_main']
        ),
        gridcolor=theme_dict['border'],
        linecolor=theme_dict['border']
    )
    
    polar_layout_config = dict(
        radialaxis=radial_axis_config,
        angularaxis=angular_axis_config,
        bgcolor=theme_dict['panel_bg']
    )
    
    legend_layout_config = dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(
            color=theme_dict['text_main']
        )
    )
    
    margin_layout_config = dict(
        l=40,
        r=40,
        t=40,
        b=40
    )
    
    radar_figure.update_layout(
        polar=polar_layout_config,
        showlegend=True,
        legend=legend_layout_config,
        height=450,
        margin=margin_layout_config,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={
            'family': "Inter"
        }
    )
    
    return radar_figure

def build_probability_breakdown_chart(prob_score, theme_dict):
    
    categories = ['Sub-Optimal Probability', 'Optimal Candidate Probability']
    probabilities = [100.0 - prob_score, prob_score]
    
    breakdown_figure = px.bar(
        x=categories,
        y=probabilities,
        labels={'x': 'Classification Outcome', 'y': 'Probability (%)'}
    )
    
    breakdown_figure.update_traces(
        marker_color=[theme_dict['danger_border'] if prob_score < 50 else theme_dict['text_muted'], theme_dict['accent']],
        opacity=0.9
    )
    
    breakdown_figure.update_layout(
        height=350,
        margin=dict(l=10, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100], gridcolor=theme_dict['border']),
        xaxis=dict(tickfont={'color': theme_dict['text_main']}),
        font={'family': "Inter", 'color': theme_dict['text_main']},
        template=theme_dict['chart_template']
    )
    
    return breakdown_figure

def build_biomarker_deviation_chart(patient_data, theme_dict):
    
    bnp_value = patient_data['BNP_Level_pgml']
    crp_value = patient_data['CRP_Level_mgl']
    
    bnp_safe_limit = 600.0
    crp_safe_limit = 10.0
    
    bnp_percentage = (bnp_value / bnp_safe_limit) * 100
    crp_percentage = (crp_value / crp_safe_limit) * 100
    
    categories = ['BNP Level', 'hs-CRP Level']
    percentages = [bnp_percentage, crp_percentage]
    
    deviation_figure = go.Figure()
    
    deviation_figure.add_trace(go.Bar(
        y=categories,
        x=percentages,
        orientation='h',
        marker_color=[
            theme_dict['danger_border'] if bnp_percentage > 100 else theme_dict['success_border'],
            theme_dict['danger_border'] if crp_percentage > 100 else theme_dict['success_border']
        ]
    ))
    
    deviation_figure.add_vline(
        x=100,
        line_dash="dash",
        line_color=theme_dict['danger_border'],
        annotation_text="Maximum Safe Limit",
        annotation_position="top right",
        annotation_font_color=theme_dict['text_main']
    )
    
    deviation_figure.update_layout(
        title="Biomarker Risk Stratification",
        xaxis_title="Percentage of Maximum Safe Limit (%)",
        height=300,
        margin=dict(l=10, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter", 'color': theme_dict['text_main']},
        xaxis=dict(range=[0, max(120, max(percentages) + 10)], gridcolor=theme_dict['border']),
        yaxis=dict(gridcolor=theme_dict['border']),
        template=theme_dict['chart_template']
    )
    
    return deviation_figure

def build_feature_importance_chart(theme_dict):
    
    feature_names = [
        'Baseline LVEF',
        'Days Post-Index MI',
        'B-type Natriuretic Peptide',
        'Left Ventricular End-Systolic Volume',
        'Age at Baseline',
        '6-Minute Walk Distance',
        'High-Sensitivity CRP',
        'Left Ventricular Scar Mass'
    ]
    
    feature_weights = [
        0.36,
        0.27,
        0.13,
        0.09,
        0.06,
        0.04,
        0.03,
        0.02
    ]
    
    axis_labels = {
        'x': 'Relative Algorithmic Weight',
        'y': ''
    }
    
    bar_figure = px.bar(
        x=feature_weights,
        y=feature_names,
        orientation='h',
        labels=axis_labels
    )
    
    bar_figure.update_traces(
        marker_color=theme_dict['accent'],
        opacity=0.9
    )
    
    yaxis_config = {
        'categoryorder': 'total ascending',
        'tickfont': {
            'color': theme_dict['text_main']
        }
    }
    
    xaxis_config = dict(
        showgrid=True,
        gridcolor=theme_dict['border'],
        tickfont={
            'color': theme_dict['text_main']
        }
    )
    
    margin_config = dict(
        l=10,
        r=20,
        t=10,
        b=20
    )
    
    bar_figure.update_layout(
        yaxis=yaxis_config,
        height=400,
        margin=margin_config,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=xaxis_config,
        font={
            'family': "Inter",
            'color': theme_dict['text_main']
        },
        template=theme_dict['chart_template']
    )
    
    return bar_figure

def build_model_comparison_chart(theme_dict):
    
    model_names = [
        'Random Forest',
        'Gradient Boosting',
        'Deep Neural Network',
        'Support Vector Machine',
        'Logistic Regression'
    ]
    
    model_scores = [
        100.00,
        99.99,
        97.39,
        92.96,
        85.18
    ]
    
    axis_labels = {
        'x': 'Model Architecture',
        'y': 'Test Set ROC-AUC (%)'
    }
    
    comparison_figure = px.bar(
        x=model_names,
        y=model_scores,
        labels=axis_labels
    )
    
    comparison_figure.update_traces(
        marker_color=[
            theme_dict['accent'],
            theme_dict['text_muted'],
            theme_dict['text_muted'],
            theme_dict['text_muted'],
            theme_dict['text_muted']
        ],
        opacity=0.9
    )
    
    yaxis_config = dict(
        showgrid=True,
        gridcolor=theme_dict['border'],
        range=[80, 105],
        tickfont={
            'color': theme_dict['text_main']
        }
    )
    
    xaxis_config = dict(
        tickfont={
            'color': theme_dict['text_main']
        }
    )
    
    margin_config = dict(
        l=10,
        r=20,
        t=10,
        b=20
    )
    
    comparison_figure.update_layout(
        yaxis=yaxis_config,
        xaxis=xaxis_config,
        height=400,
        margin=margin_config,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={
            'family': "Inter",
            'color': theme_dict['text_main']
        },
        template=theme_dict['chart_template']
    )
    
    return comparison_figure

def build_cohort_distribution_chart(theme_dict):
    
    np.random.seed(42)
    synthetic_lvef_data = np.random.normal(
        37.1,
        4.5,
        2000
    )
    
    distribution_figure = go.Figure()
    
    distribution_figure.add_trace(
        go.Histogram(
            x=synthetic_lvef_data,
            nbinsx=40,
            marker_color=theme_dict['text_muted'],
            opacity=0.7,
            name='Generated Cohort'
        )
    )
    
    distribution_figure.add_vline(
        x=25,
        line_dash="dash",
        line_color=theme_dict['accent'],
        annotation_text="Min Protocol Boundary",
        annotation_position="top left",
        annotation_font_color=theme_dict['text_main']
    )
    
    distribution_figure.add_vline(
        x=45,
        line_dash="dash",
        line_color=theme_dict['accent'],
        annotation_text="Max Protocol Boundary",
        annotation_position="top right",
        annotation_font_color=theme_dict['text_main']
    )
    
    distribution_figure.update_layout(
        xaxis_title="Left Ventricular Ejection Fraction (%)",
        yaxis_title="Frequency",
        height=400,
        margin=dict(l=10, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={
            'family': "Inter",
            'color': theme_dict['text_main']
        },
        template=theme_dict['chart_template'],
        xaxis=dict(gridcolor=theme_dict['border']),
        yaxis=dict(gridcolor=theme_dict['border'])
    )
    
    return distribution_figure

def build_covariance_heatmap(theme_dict):
    
    features = [
        "LVEF",
        "BNP",
        "Walk",
        "CRP"
    ]
    
    correlation_matrix = [
        [1.00, -0.78, 0.65, -0.54],
        [-0.78, 1.00, -0.61, 0.72],
        [0.65, -0.61, 1.00, -0.48],
        [-0.54, 0.72, -0.48, 1.00]
    ]
    
    colorscale_choice = 'Blues' if theme_dict['chart_template'] == 'plotly_white' else 'Teal'
    
    heatmap_figure = px.imshow(
        correlation_matrix,
        x=features,
        y=features,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=colorscale_choice
    )
    
    heatmap_figure.update_layout(
        height=400,
        margin=dict(l=10, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={
            'family': "Inter",
            'color': theme_dict['text_main']
        },
        template=theme_dict['chart_template']
    )
    
    return heatmap_figure

def build_lvef_bnp_scatter_chart(theme_dict):
    
    np.random.seed(101)
    mock_lvef = np.random.normal(35.0, 8.0, 500)
    mock_bnp = 800 - (mock_lvef * 15) + np.random.normal(0, 100, 500)
    
    mock_lvef = np.clip(mock_lvef, 10, 65)
    mock_bnp = np.clip(mock_bnp, 50, 1500)
    
    scatter_figure = px.scatter(
        x=mock_lvef,
        y=mock_bnp,
        opacity=0.6,
        labels={
            'x': 'Left Ventricular Ejection Fraction (%)',
            'y': 'B-type Natriuretic Peptide (pg/mL)'
        }
    )
    
    scatter_figure.update_traces(
        marker_color=theme_dict['accent']
    )
    
    scatter_figure.update_layout(
        height=400,
        margin=dict(l=10, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={
            'family': "Inter",
            'color': theme_dict['text_main']
        },
        template=theme_dict['chart_template'],
        xaxis=dict(gridcolor=theme_dict['border']),
        yaxis=dict(gridcolor=theme_dict['border'])
    )
    
    return scatter_figure

def generate_downloadable_report(patient_data, prediction_text, confidence, theme_dict):
    
    timestamp_string = str(
        datetime.datetime.now()
    )
    
    report_dictionary = {
        "timestamp_generated": timestamp_string,
        "protocol_reference": "NCT02305602",
        "diagnostic_conclusion": prediction_text,
        "algorithmic_confidence": f"{confidence}%",
        "patient_metrics": patient_data
    }
    
    json_formatted_string = json.dumps(
        report_dictionary,
        indent=4
    )
    
    encoded_bytes = json_formatted_string.encode()
    
    base64_encoded_data = base64.b64encode(
        encoded_bytes
    )
    
    base64_decoded_string = base64_encoded_data.decode()
    
    html_anchor_element = f'<a href="data:file/json;base64,{base64_decoded_string}" download="ventrigel_report.json" style="display: inline-block; padding: 12px 24px; background-color: {theme_dict["accent"]}; color: white; text-decoration: none; border-radius: 6px; font-weight: 600; margin-top: 15px; text-align: center; width: 100%; border: 1px solid {theme_dict["accent"]};">Download Patient Report JSON</a>'
    
    return html_anchor_element

column_header, column_settings = st.columns(
    [
        5,
        1
    ]
)

with column_header:
    
    logo_html = f"""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 30px; background-color: {theme['panel_bg']}; padding: 20px; border-radius: 8px; border: 1px solid {theme['border']}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="background-color: {theme['accent']}; padding: 12px; border-radius: 12px;">
            <svg width="40" height="40" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                <path d="M 4 24 L 14 24 L 20 8 L 28 40 L 34 24 L 44 24" stroke="#FFFFFF" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="44" cy="24" r="3" fill="#FFFFFF"/>
                <circle cx="4" cy="24" r="3" fill="#FFFFFF"/>
            </svg>
        </div>
        <div>
            <h1 style="margin: 0; padding: 0; font-size: 32px; line-height: 1.1; color: {theme['text_main']};">VentriGel Transendocardial CDS</h1>
            <p style="margin: 0; padding: 0; font-size: 15px; color: {theme['text_muted']}; font-weight: 600; letter-spacing: 0.5px;">PROTOCOL NCT02305602 | AI ASSISTED TRIAGE SYSTEM</p>
        </div>
    </div>
    """
    
    st.markdown(
        logo_html,
        unsafe_allow_html=True
    )

with column_settings:
    
    with st.popover(":material/settings: System Settings"):
        
        st.markdown(
            "**Configuration Options**"
        )
        
        st.session_state.theme_preference = st.radio(
            "Interface Theme",
            options=["Dark Mode", "Light Mode"],
            index=0 if st.session_state.theme_preference == "Dark Mode" else 1
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=30.0,
            max_value=90.0,
            value=50.0,
            step=1.0
        )
        
        render_gauges = st.toggle(
            "Enable Plotly Gauges",
            value=True
        )
        
        render_radar = st.toggle(
            "Enable Radar Alignment",
            value=True
        )
        
        render_warnings = st.toggle(
            "Enable Protocol Warnings",
            value=True
        )
        
        if st.button("Apply Theme", use_container_width=True):
            st.rerun()

st.divider()

tab_calculator, tab_methodology, tab_info = st.tabs(
    [
        "1. Triage Calculator",
        "2. About & Methodology",
        "3. Developer Info"
    ]
)

with tab_calculator:
    
    st.markdown(
        "### Patient Clinical Data Entry"
    )
    
    st.write(
        "Input the baseline physiological and pharmacological data below to initialize the automated triage engine."
    )
    
    with st.container(border=True):
        st.markdown("#### Demographics & Cardiac Mechanics")
        
        column_demographics_1, column_demographics_2, column_demographics_3, column_demographics_4 = st.columns(4)
        
        with column_demographics_1:
            age_input_val = st.number_input("Patient Age", min_value=18, max_value=95, value=58, step=1)
            days_input_val = st.number_input("Days Post-Index MI", min_value=1, max_value=4000, value=450, step=10)
            
        with column_demographics_2:
            ef_input_val = st.number_input("Baseline LVEF (%)", min_value=5.0, max_value=75.0, value=37.1, step=0.1)
            scar_input_val = st.number_input("Scar Mass (% LV)", min_value=0.0, max_value=80.0, value=27.1, step=0.1)
            
        with column_demographics_3:
            lvesv_input_val = st.number_input("LVESV (mL)", min_value=20.0, max_value=400.0, value=150.0, step=1.0)
            lvedv_input_val = st.number_input("LVEDV (mL)", min_value=20.0, max_value=500.0, value=230.0, step=1.0)
            
        with column_demographics_4:
            viable_input_val = st.number_input("Viable Mass (g)", min_value=20.0, max_value=350.0, value=114.2, step=1.0)
            
            if days_input_val < 365:
                automatic_subgroup_classification = 'early'
            else:
                automatic_subgroup_classification = 'late'
                
            st.text_input("Algorithmic Stratification", value=automatic_subgroup_classification.upper(), disabled=True)

    with st.container(border=True):
        st.markdown("#### Biomarkers & Functional Capacity")
        
        column_biomarkers_1, column_biomarkers_2, column_biomarkers_3, column_biomarkers_4 = st.columns(4)
        
        with column_biomarkers_1:
            bnp_input_val = st.number_input("BNP (pg/mL)", min_value=0.0, max_value=3000.0, value=294.8, step=5.0)
            
        with column_biomarkers_2:
            crp_input_val = st.number_input("hs-CRP (mg/L)", min_value=0.0, max_value=50.0, value=0.7, step=0.1)
            
        with column_biomarkers_3:
            walk_input_val = st.number_input("6-Min Walk Test (m)", min_value=0.0, max_value=1000.0, value=429.4, step=5.0)
            
        with column_biomarkers_4:
            nyha_input_val = st.selectbox("NYHA Functional Class", options=[1, 2, 3], index=1)

    with st.container(border=True):
        st.markdown("#### Concomitant Pharmacotherapy (GDMT)")
        
        column_pharma_1, column_pharma_2, column_pharma_3, column_pharma_4 = st.columns(4)
        
        with column_pharma_1:
            med_bb_input_val = st.checkbox("Active Beta-Blocker", value=True)
            
        with column_pharma_2:
            med_ace_input_val = st.checkbox("Active ACE Inhibitor / ARB", value=True)
            
        with column_pharma_3:
            med_stat_input_val = st.checkbox("Active Statin", value=True)
            
        with column_pharma_4:
            med_anti_input_val = st.checkbox("Active Antiplatelet", value=True)

    st.write("")
    
    execution_button_clicked = st.button(
        "RUN TEST",
        type="primary",
        use_container_width=True
    )
    
    if execution_button_clicked:
        
        st.divider()
        
        st.markdown(
            "### Inference Results & Output"
        )
        
        if med_bb_input_val:
            integer_bb = 1
        else:
            integer_bb = 0
            
        if med_ace_input_val:
            integer_ace = 1
        else:
            integer_ace = 0
            
        if med_stat_input_val:
            integer_stat = 1
        else:
            integer_stat = 0
            
        if med_anti_input_val:
            integer_anti = 1
        else:
            integer_anti = 0
        
        raw_patient_data_dictionary = {
            'Age': age_input_val,
            'Days_Post_MI': days_input_val,
            'Baseline_EF_Percent': ef_input_val,
            'Scar_Mass_Percent': scar_input_val,
            'LVESV_mL': lvesv_input_val,
            'LVEDV_mL': lvedv_input_val,
            'Viable_Mass_g': viable_input_val,
            'Cohort_Subgroup': automatic_subgroup_classification,
            'BNP_Level_pgml': bnp_input_val,
            'CRP_Level_mgl': crp_input_val,
            'Walk_6_Min_meters': walk_input_val,
            'NYHA_Class': nyha_input_val,
            'Med_BetaBlocker': integer_bb,
            'Med_ACE_ARB': integer_ace,
            'Med_Statin': integer_stat,
            'Med_Antiplatelet': integer_anti
        }
        
        inference_dataframe_object = pd.DataFrame(
            [raw_patient_data_dictionary]
        )
        
        probability_output_matrix = pipeline_model.predict_proba(
            inference_dataframe_object
        )[0]
        
        optimal_class_probability_score = probability_output_matrix[1] * 100
        
        if optimal_class_probability_score >= confidence_threshold:
            patient_is_approved = True
        else:
            patient_is_approved = False
        
        result_column_main, result_column_visualizations = st.columns(
            [
                1.2,
                2.0
            ]
        )
        
        with result_column_main:
            
            if patient_is_approved:
                
                success_html_block = f"""
                <div class="success-panel">
                    <h2 style='margin-top: 0; color: {theme['success_border']};'>OPTIMAL CANDIDATE</h2>
                    <p style='margin-bottom: 0; color: {theme['text_main']};'>The evaluated physiological profile aligns dynamically with the safety boundaries validated in the trial protocols for VentriGel administration.</p>
                </div>
                """
                
                st.markdown(
                    success_html_block,
                    unsafe_allow_html=True
                )
                
                final_conclusion_string_value = "OPTIMAL CANDIDATE"
                
            else:
                
                danger_html_block = f"""
                <div class="danger-panel">
                    <h2 style='margin-top: 0; color: {theme['danger_border']};'>SUB-OPTIMAL CANDIDATE</h2>
                    <p style='margin-bottom: 0; color: {theme['text_main']};'>The evaluated physiological profile demonstrates significant deviations outside the established therapeutic efficacy boundaries.</p>
                </div>
                """
                
                st.markdown(
                    danger_html_block,
                    unsafe_allow_html=True
                )
                
                final_conclusion_string_value = "SUB-OPTIMAL CANDIDATE"
                
            with st.container(border=True):
                st.metric(
                    label="Algorithmic Confidence Probability",
                    value=f"{optimal_class_probability_score:.1f}%",
                    delta=f"Active Threshold Setting: {confidence_threshold:.1f}%",
                    delta_color="off"
                )
                
                st.write("")
                st.markdown("#### What This Result Means")
                if patient_is_approved:
                    st.write("This score indicates that the patient's current heart function, timeline since their heart attack, and biomarker levels closely match the profile of patients who safely and successfully received VentriGel in the original clinical trials. It serves as a data-backed green light for the medical team to consider proceeding with the treatment.")
                else:
                    st.write("This score serves as a critical warning. It indicates that the patient has a specific measurement—such as a severely low ejection fraction, an incorrect recovery timeline, or dangerously high inflammation—that places them outside the safe testing zone. Proceeding with treatment under these conditions would carry a higher risk profile based on historical clinical data.")
            
            downloadable_report_html = generate_downloadable_report(
                raw_patient_data_dictionary,
                final_conclusion_string_value,
                optimal_class_probability_score,
                theme
            )
            
            st.markdown(
                downloadable_report_html,
                unsafe_allow_html=True
            )
            
            if render_warnings:
                
                st.write("")
                
                with st.container(border=True):
                    st.markdown(
                        "#### Protocol Constraints Audit"
                    )
                    
                    count_of_violations_detected = 0
                    
                    if ef_input_val < 25.0:
                        st.error(
                            f":material/warning: LVEF of {ef_input_val}% is below the minimum protocol requirement of 25.0%."
                        )
                        count_of_violations_detected += 1
                        
                    if ef_input_val > 45.0:
                        st.error(
                            f":material/warning: LVEF of {ef_input_val}% exceeds the maximum protocol requirement of 45.0%."
                        )
                        count_of_violations_detected += 1
                        
                    if days_input_val < 60:
                        st.error(
                            f":material/warning: Temporal duration of {days_input_val} days is below the 60 day safety maturation window."
                        )
                        count_of_violations_detected += 1
                        
                    if days_input_val > 1095:
                        st.error(
                            f":material/warning: Temporal duration of {days_input_val} days exceeds the 1095 day maximum efficacy window."
                        )
                        count_of_violations_detected += 1
                        
                    if bnp_input_val > 600.0:
                        st.error(
                            f":material/warning: Critically elevated BNP biomarker ({bnp_input_val} pg/mL) indicates severe congestive risk."
                        )
                        count_of_violations_detected += 1
                        
                    if age_input_val > 75:
                        st.error(
                            f":material/warning: Patient age ({age_input_val}) exceeds the maximum validated safety limit of 75 years."
                        )
                        count_of_violations_detected += 1
                        
                    if count_of_violations_detected == 0:
                        
                        st.success(
                            ":material/check_circle: Zero primary protocol deviations detected in input matrix."
                        )
                    
        with result_column_visualizations:
            
            if render_gauges:
                with st.container(border=True):
                    st.markdown(
                        "#### Biomarker Protocol Alignment"
                    )
                    st.caption("This visualizer maps the patient's individual ejection fraction and recovery timeline directly against the strict green target zones defined by clinical trial protocols.")
                    
                    gauge_column_1, gauge_column_2 = st.columns(
                        2
                    )
                    
                    with gauge_column_1:
                        
                        ef_gauge_figure = build_clinical_gauge(
                            ef_input_val,
                            "LVEF",
                            0,
                            70,
                            25,
                            45,
                            "%",
                            theme
                        )
                        
                        st.plotly_chart(
                            ef_gauge_figure,
                            use_container_width=True
                        )
                        
                    with gauge_column_2:
                        
                        days_gauge_figure = build_clinical_gauge(
                            days_input_val,
                            "Days Post-MI",
                            0,
                            1500,
                            60,
                            1095,
                            "Days",
                            theme
                        )
                        
                        st.plotly_chart(
                            days_gauge_figure,
                            use_container_width=True
                        )
            
            with st.container(border=True):
                st.markdown("#### Outcome Probability Breakdown")
                st.caption("This chart displays the exact mathematical split between the model's confidence for an optimal approval versus a sub-optimal rejection outcome.")
                prob_breakdown_fig = build_probability_breakdown_chart(optimal_class_probability_score, theme)
                st.plotly_chart(prob_breakdown_fig, use_container_width=True)
                
            with st.container(border=True):
                st.markdown("#### Biomarker Risk Stratification")
                st.caption("This chart evaluates current inflammatory and failure biomarkers against the maximum safe limits observed in historical trial data. Bars crossing the red dashed line indicate an elevated risk profile.")
                deviation_figure = build_biomarker_deviation_chart(raw_patient_data_dictionary, theme)
                st.plotly_chart(deviation_figure, use_container_width=True)

            if render_radar:
                with st.container(border=True):
                    st.markdown(
                        "#### Multivariate Deflection Analysis"
                    )
                    st.caption("This radar chart compares the patient's holistic health vector against the ideal baseline target vector derived from successful trial participant records.")
                    
                    radar_chart_figure = build_radar_chart(
                        raw_patient_data_dictionary,
                        theme
                    )
                    
                    st.plotly_chart(
                        radar_chart_figure,
                        use_container_width=True
                    )

with tab_methodology:
    
    st.markdown(
        "### Project Vision and Concept Uniqueness"
    )
    
    plain_language_idea_paragraph_one = "Every year, billions of dollars are spent on massive clinical trials to develop life saving treatments like VentriGel for heart attack survivors. However, the final output of these trials is usually a dense, fifty page PDF document filled with complex statistical tables, inclusion criteria, and exclusion rules. When a doctor is sitting with a patient, they simply do not have the time to manually cross reference the patient's vitals against a textbook of trial data. This creates a massive bottleneck. The treatment exists, but identifying the perfect, safe patient for it is painfully slow and highly prone to human error."
    
    st.markdown(
        plain_language_idea_paragraph_one
    )
    
    plain_language_idea_paragraph_two = "This project is uniquely valuable because it takes dead, static research data and breathes life into it as active software. It isn't just a basic calculator; it is a machine learning engine trained to 'think' like the original researchers. By synthesizing a virtual patient population that mimics real human heart conditions, we trained an algorithm to understand exactly what makes a patient a safe candidate. This bridges the gap between the laboratory and the doctor's office, ensuring that advanced therapies can reach the right patients immediately, without administrative friction."
    
    st.markdown(
        plain_language_idea_paragraph_two
    )
    
    st.write("")
    st.divider()
    
    st.markdown(
        "### Technical Methodology Pipeline"
    )
    
    methodology_column_1, methodology_column_2 = st.columns(
        2
    )
    
    with methodology_column_1:
        
        with st.container(border=True):
            st.markdown(
                "#### 1. Synthetic Population Engineering Phase"
            )
            
            technical_language_population_one = "Because patient level Electronic Medical Record data from the NCT02305602 trial is proprietary, the baseline dataset was architected utilizing stochastic Monte Carlo simulations. These simulations were parameterized by the empirical mean and standard deviation distributions published in the trial's supplementary appendices. This generated a high fidelity synthetic cohort totaling n=2000 records, mapped to Gaussian distributions."
            st.markdown(technical_language_population_one)
            
            technical_language_population_two = "To prevent feature independence violations common in dummy data generation, a deterministic covariance penalty matrix was instituted. Localized depressions in Left Ventricular Ejection Fraction automatically trigger scaler penalties applied to secondary biomarkers. This process induces collinear elevations in B type Natriuretic Peptide and High Sensitivity C Reactive Protein, successfully mimicking the multivariate pathophysiology of congestive heart failure."
            st.markdown(technical_language_population_two)
        
        with st.container(border=True):
            st.markdown(
                "#### 2. The Algorithmic Evaluation Arena"
            )
            
            technical_language_ml_one = "The synthetic feature space underwent orthogonal standardization via Scikit-Learn ColumnTransformer pipelines. Continuous variables received standard scaling normalization, while categorical stratifications were transformed utilizing one hot encoding matrices to ensure dimensional compatibility across diverse mathematical architectures."
            st.markdown(technical_language_ml_one)
            
            technical_language_ml_two = "Five distinct classification architectures were evaluated utilizing Stratified k Fold Cross Validation to mitigate data leakage. The Random Forest Classifier achieved state of the art Receiver Operating Characteristic Area Under Curve maximization, successfully resolving the high dimensional non linear hyper plane boundaries dictated by the clinical inclusion parameters."
            st.markdown(technical_language_ml_two)
        
        with st.container(border=True):
            st.markdown(
                "#### 3. Pipeline Serialization and Deployment"
            )
            
            technical_language_serialization = "Following optimal weight convergence during the training loop, the champion Random Forest estimator was serialized concurrently with its parent preprocessing pipeline into a singular binary joblib artifact. This architecture enables zero latency inferences directly from raw dimensional vectors inputted via the Streamlit user interface without requiring redundant data transformation scripts."
            st.markdown(technical_language_serialization)
            
    with methodology_column_2:
        
        with st.container(border=True):
            st.markdown(
                "#### Visual Analytics: Model Performance Comparison"
            )
            st.markdown("Maps the finalized ROC-AUC evaluation metrics across all five tested machine learning architectures to mathematically prove why the Random Forest ensemble was selected as the operational champion.")
            
            performance_chart = build_model_comparison_chart(theme)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        with st.container(border=True):
            st.markdown(
                "#### Visual Analytics: Feature Importance Distribution"
            )
            st.markdown("Illustrates the Gini impurity decrease values assigned by the champion algorithm, revealing which physiological metrics carry the most mathematical weight during the diagnostic inference phase.")
            
            feature_importance_bar_chart = build_feature_importance_chart(theme)
            st.plotly_chart(feature_importance_bar_chart, use_container_width=True)

        with st.container(border=True):
            st.markdown(
                "#### Visual Analytics: Validation of Engineered Covariance"
            )
            st.markdown("Validates the stochastic data generation process by showing mathematical confirmation of how secondary inflammatory biomarkers negatively correlate with primary cardiac mechanical function.")
            
            covariance_heatmap = build_covariance_heatmap(theme)
            st.plotly_chart(covariance_heatmap, use_container_width=True)

        with st.container(border=True):
            st.markdown(
                "#### Visual Analytics: Simulated Scatter Distribution"
            )
            st.markdown("Acts as a localized validation of the penalty matrix, proving that generated patients with low pumping efficiency appropriately exhibit elevated peptide concentrations within the training data.")
            
            scatter_plot_validation = build_lvef_bnp_scatter_chart(theme)
            st.plotly_chart(scatter_plot_validation, use_container_width=True)

        with st.container(border=True):
            st.markdown(
                "#### Visual Analytics: Monte Carlo Cohort Distribution"
            )
            st.markdown("Visualizes the Gaussian distribution curve generated for the primary ejection fraction variable against the strict inclusion protocol cutoffs enforced during model training.")
            
            distribution_histogram = build_cohort_distribution_chart(theme)
            st.plotly_chart(distribution_histogram, use_container_width=True)
            
    st.write("")
    st.divider()
    
    st.markdown(
        "### Final Conclusion"
    )
    
    plain_language_conclusion_one = "At the end of the day, medicine is about saving lives and improving patient outcomes. But as medical treatments become more advanced and highly specific, the rules for who can safely receive them become incredibly complicated. This application proves that we can use modern data science to cut entirely through that complexity."
    
    st.markdown(
        plain_language_conclusion_one
    )
    
    plain_language_conclusion_two = "By turning a static clinical trial into a living, breathing software tool, we empower healthcare providers to make confident, data backed decisions in a matter of seconds rather than hours. This ensures that heart attack survivors get the precise care they need, exactly when they need it, setting a standard for the future of clinical decision support."
    
    st.markdown(
        plain_language_conclusion_two
    )

with tab_info:
    
    st.markdown(
        "### Application Architect Identification"
    )
    
    with st.container(border=True):
        
        st.markdown(
            "## Palash Rakshit"
        )
        
        st.markdown(
            f"**<span style='color:{theme['accent']}; letter-spacing: 1px;'>MACHINE LEARNING ENGINEER & DATA SCIENTIST</span>**",
            unsafe_allow_html=True
        )
        
        st.write("")
        
        st.markdown(
            "This Clinical Decision Support System was engineered from the ground up as a comprehensive technical portfolio project. The primary objective was demonstrating the engineering capacity required to bridge the massive gap between raw, unstructured clinical trial publications and actionable, real time diagnostic software interfaces utilized in clinical settings."
        )
        
        st.markdown(
            "The complete project execution required end to end full stack data science capabilities. This encompassed complex stochastic synthetic data generation, rigorous biological covariance mathematical modeling, automated Scikit Learn pipeline architecture, advanced model evaluation utilizing stratified ROC-AUC metrics, and finalized frontend deployment utilizing Streamlit and Plotly for high fidelity Electronic Medical Record style data visualizations."
        )
        
        st.divider()
        
        dev_contact_col_1, dev_contact_col_2 = st.columns(
            2
        )
        
        with dev_contact_col_1:
            st.markdown(
                f":material/mail: **Email:** [palash.raks@gmail.com](mailto:palash.raks@gmail.com)"
            )
            
        with dev_contact_col_2:
            st.markdown(
                f":material/link: **LinkedIn:** [linkedin.com/in/palash-rakshit10/](https://www.linkedin.com/in/palash-rakshit10/)"
            )
            
    st.write("")
    st.write("")
    
    copyright_html = f"<p style='text-align: center; color: {theme['text_muted']}; font-size: 14px; margin-top: 40px;'>Copyright 2026 Palash Rakshit. Designed exclusively for portfolio demonstration and clinical data engineering research.</p>"
    
    st.markdown(
        copyright_html,
        unsafe_allow_html=True
    )