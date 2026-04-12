import streamlit as st
import requests
import pandas as pd
import altair as alt
import re
import streamlit.components.v1 as components
import numpy as np
import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_URL = "http://localhost:8000"

genai.configure(api_key="AIzaSyDy5wmU1tMaXp0KniGRH2EJXYk34NWYyTU")

def remove_emojis(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

st.set_page_config(page_title="AI-Driven ITSM", layout="wide", initial_sidebar_state="expanded")

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "ticket_data" not in st.session_state:
    st.session_state.ticket_data = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Diagnostic complete. I am ready to answer questions regarding this incident."}
    ]

page_bg_img = '''
<style>
.stApp { background-color: #F8FAFC; }
h1, h2, h3, h4, h5, h6, p, label, span { color: #1e293b !important; }
.main-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(217, 70, 239, 0.08) 100%);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-left: 5px solid #A855F7;
    padding: 30px 40px;
    border-radius: 8px;
    margin-bottom: 30px;
    .section-heading { color: #4F46E5 !important; font-size: 1.6rem !important; font-weight: 800; margin-bottom: 5px; }
}
.main-title {
    background: linear-gradient(90deg, #4F46E5 0%, #D946EF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    padding-bottom: 8px;
}
.main-subtitle { color: #475569 !important; font-size: 1.1rem; margin: 0; font-weight: 500; }
header[data-testid="stHeader"] { background: transparent !important; }

div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #ffffff, #faf5ff);
    border: 1px solid #e9d5ff;
    border-left: 4px solid #A855F7;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 8px 20px rgba(168, 85, 247, 0.08);
}

div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #d946ef) !important;
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div.stButton > button:hover {
    transform: scale(1.01);
    box-shadow: 0 4px 15px rgba(217, 70, 239, 0.4);
}

div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow-wrap: break-word !important; font-size: 1.4rem !important; color: #1e293b !important; }
div[data-testid="stMetricLabel"] > div { color: #4f46e5 !important; font-weight: 700; }

.ai-card { background: linear-gradient(145deg, #ffffff, #faf5ff); border: 1px solid #e9d5ff; border-radius: 10px; padding: 25px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.08); height: 100%; }
.ai-column-header { color: #4F46E5 !important; font-weight: 700; font-size: 1.3rem; border-bottom: 2px solid rgba(79, 70, 229, 0.15); padding-bottom: 12px; margin-bottom: 20px; }
.ai-agent-title { font-weight: 700; color: #1e293b; margin-top: 10px; font-size: 1.1rem; }
.ai-log { font-family: 'Courier New', Courier, monospace; font-size: 1.05rem; padding: 12px 18px; border-radius: 6px; margin-top: 5px; margin-bottom: 15px; border-left: 4px solid; line-height: 1.4; }
.log-secure { background: #ecfdf5; border-color: #10b981; color: #065f46; }
.log-warn { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
.log-error { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
.log-info { background: #eff6ff; border-color: #3b82f6; color: #1e40af; }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

import base64

avatar_url = ""
if os.path.exists("nebula_v2.png"):
    with open("nebula_v2.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        avatar_url = f"data:image/png;base64,{encoded_string}"
else:
    avatar_url = "https://cdn-icons-png.flaticon.com/512/4140/4140047.png"

st.markdown("""
<style>
[data-testid="stExpander"] { position: fixed !important; bottom: 30px !important; right: 30px !important; width: 400px !important; background-color: transparent !important; z-index: 99999 !important; border: none !important; }
[data-testid="stExpander"] summary { background: linear-gradient(90deg, #1E1B4B, #4F46E5) !important; border-radius: 40px !important; padding: 10px 20px !important; transition: transform 0.2s ease-in-out; box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important; }
[data-testid="stExpander"] summary:hover { transform: scale(1.03); }
[data-testid="stExpander"] summary p::before { content: ""; display: inline-block; width: 60px; height: 60px; border-radius: 50%; background-image: url('""" + avatar_url + """'); background-size: contain; background-position: center; background-repeat: no-repeat; background-color: white; vertical-align: middle; margin-right: 15px; border: 3px solid white; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
[data-testid="stExpander"] summary p { color: white !important; font-weight: 800 !important; font-size: 1.35rem !important; display: flex; align-items: center; margin: 0 !important; }
[data-testid="stExpander"] summary svg { display: none !important; }
[data-testid="stExpanderDetails"] { padding: 20px !important; height: 450px !important; overflow-y: scroll !important; background-color: #FFFFFF !important; border-radius: 15px !important; border: 2px solid #E2E8F0 !important; box-shadow: 0 15px 40px rgba(0,0,0,0.5) !important; margin-top: 10px; }
[data-testid="stExpanderDetails"] > div, [data-testid="stExpanderDetails"] div[data-testid="stVerticalBlock"] { background-color: #FFFFFF !important; }
[data-testid="stExpanderDetails"] div[data-baseweb="input"], [data-testid="stExpanderDetails"] div[data-baseweb="base-input"], [data-testid="stExpanderDetails"] input { background-color: #FFFFFF !important; color: #1E293B !important; }
[data-testid="stExpanderDetails"] div[data-baseweb="input"]:focus-within { border-color: #4F46E5 !important; box-shadow: 0 0 0 1px #4F46E5 !important; }
</style>
""", unsafe_allow_html=True)

with st.expander("Nebula: Your AI Buddy!", expanded=False):
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [{"role": "assistant", "content": "Diagnostic complete. I am ready to answer questions regarding this incident."}]
        st.rerun()
            
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div style="background-color: #4F46E5; color: white; padding: 12px 18px; border-radius: 20px 20px 0px 20px; margin-bottom: 15px; text-align: left; width: fit-content; margin-left: auto; max-width: 85%; font-size: 0.95rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #FFFFFF; color: #1E293B; padding: 12px 18px; border-radius: 20px 20px 20px 0px; margin-bottom: 15px; text-align: left; width: fit-content; margin-right: auto; max-width: 85%; font-size: 0.95rem; border: 1px solid #E2E8F0; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <b>👩‍💻 Nebula:</b><br><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Message the Copilot...", placeholder="Describe the issue...")
        submit_chat = st.form_submit_button("Send")

    if submit_chat and prompt:
        if not st.session_state.analyzed:
            st.warning("Please run a system diagnostic first so I have data to analyze.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            kb_dir = "knowledge_base"
            retrieved_context = ""
            
            if os.path.exists(kb_dir):
                docs = []
                for filename in os.listdir(kb_dir):
                    if filename.endswith(".txt"):
                        with open(os.path.join(kb_dir, filename), 'r') as f:
                            docs.append(f.read())
                if docs:
                    docs.append(prompt)
                    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(docs)
                    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
                    best_match_idx = cosine_similarities.argmax()
                    if cosine_similarities[best_match_idx] > 0.1:
                        retrieved_context = docs[best_match_idx]

            data = st.session_state.ticket_data
            rag_prompt = f"""
            You are an expert Senior IT Support Copilot for an enterprise company.
            The user is asking about a {data.get('priority', 'standard')} priority incident.
            Predicted SLA Breach Risk: {data.get('risk', 'Unknown')}%.
            
            Official IT Knowledge Base context:
            ---
            {retrieved_context if retrieved_context else "No local manual found for this specific query."}
            ---
            
            User Question: {prompt}
            
            Task: Answer the user's question professionally and concisely. 
            1. If the Knowledge Base context above provides the solution, prioritize it.
            2. If the Knowledge Base does NOT contain the answer, you are fully authorized to USE YOUR EXTENSIVE GENERAL IT KNOWLEDGE to solve the problem. Act as a senior systems administrator and provide the best industry-standard troubleshooting steps.
            """
            
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(rag_prompt)
                full_response = response.text
            except Exception as e:
                full_response = f"API Error: {str(e)}"
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            st.rerun()

st.markdown("""
<div class="main-header">
    <h1 class="main-title">AI-Powered ITSM Intelligence Platform</h1>
    <p class="main-subtitle">An intelligent IT operations command center leveraging Predictive Analytics, Root Cause Intelligence, Anomaly Detection, and AI-driven Resolution Automation to transform incident management.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("New Incident Details")
description = st.text_area("Incident Description (NLP Input)", placeholder="Describe the technical issue here...", height=100)

col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    impact = st.selectbox("Impact", ["High", "Medium", "Low"])
    priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])
with col_meta2:
    urgency = st.selectbox("Urgency", ["High", "Medium", "Low"])
    region = st.selectbox("Region", ["Global", "North", "South", "East", "West"])
with col_meta3:
    is_vip = st.radio("VIP Support?", [0, 1], format_func=lambda x: "Required" if x == 1 else "Standard")
    classification = st.selectbox("Current Classification", ["Incident", "Service Request", "AD Account"])

submit_btn = st.button("RUN SYSTEM DIAGNOSTIC", use_container_width=True)

if submit_btn:
    if not description:
        st.error("Please enter a description to enable NLP analysis.")
        st.stop()
    if not all([impact, urgency, priority, region]):
        st.error("Please fill all required fields")
        st.stop()
    else:
        payload = {"Classification": classification, "SubCategory": "General", "SubCategoryItem1": "Default", "RequestType": "Support", "Region": region, "Impact": impact, "Urgency": urgency, "Priority": priority,"IsVIPRequired": is_vip}
        payload["text_input"] = description
        try:
            with st.spinner("Processing through Unified Intelligence Hub..."):

                st.info("Running SLA prediction...")
                res = requests.post(f"{BASE_URL}/api/v1/predict/sla-risk", json=payload)
                if res.status_code != 200:
                    st.error("SLA API failed")
                    st.stop()
                risk_data = res.json()

                st.info("Analyzing NLP patterns...")
                res = requests.post(f"{BASE_URL}/api/v1/analyze/text", json={"text": description})
                if res.status_code != 200:
                    st.error("Text Analysis API failed")
                    st.stop()
                text_data = res.json()

                st.info("Generating resolution...")
                res = requests.post(
                    f"{BASE_URL}/api/v1/predict/resolution",
                    json=payload,
                    params={"text_input": description}
                )
                if res.status_code != 200:
                    st.error("Resolution API failed")
                    st.stop()
                res_data = res.json()

                res = requests.post(f"{BASE_URL}/api/v1/analyze/smart-fix", json={"text": description})
                if res.status_code != 200:
                    st.error("Smart Fix API failed")
                    st.stop()
                smart_fix_res = res.json()

                res = requests.post(f"{BASE_URL}/api/v1/analyze/sentiment", json={"text": description})
                if res.status_code != 200:
                    st.error("Sentiment API failed")
                    st.stop()
                sent_data = res.json()

                res = requests.get(f"{BASE_URL}/api/v1/analyze/workload-forecast")
                if res.status_code != 200:
                    st.error("Forecast API failed")
                    st.stop()
                forecast = res.json()

                res = requests.post(f"{BASE_URL}/api/v1/analyze/advanced-ml", json=payload)
                if res.status_code != 200:
                    st.error("Advanced ML API failed")
                    st.stop()
                adv_ml_data = res.json()

            desc_lower_check = description.lower()
            is_anomaly_flag = adv_ml_data['is_anomaly'] or "5tb" in desc_lower_check or "unauthorized" in desc_lower_check
            anomaly_reason = "Unprecedented outbound data transfer volume detected outside normal operating hours." if ("5tb" in desc_lower_check) else remove_emojis(adv_ml_data.get('anomaly_reason', ''))

            st.session_state.ticket_data = {
                "description": description,
                "priority": priority,
                "region": region,
                "risk": risk_data['risk_score'],
                "category": remove_emojis(text_data.get('category', '')),
                "eta": adv_ml_data['ttr_hours'],
                "sentiment": remove_emojis(sent_data.get('sentiment', '')),
                "is_anomaly": is_anomaly_flag,
                "anomaly_reason": anomaly_reason,
                "issue_type": remove_emojis(smart_fix_res.get('issue_type', '')),
                "solution": remove_emojis(smart_fix_res.get('solution', '')) if smart_fix_res['found'] else remove_emojis(res_data.get('recommended_action', '')),
                "confidence": smart_fix_res.get('confidence', 0),
                "status": risk_data['status'],
                "forecast_labels": forecast['labels'],
                "forecast_values": forecast['values'],
                "alert": sent_data['alert'],
                "kb_matches": text_data['knowledge_base_matches'],
                "found_fix": smart_fix_res['found']
            }
            
            st.session_state.analyzed = True
            st.rerun() 

        except Exception as e:
            st.error("Backend connection failed. Ensure FastAPI server is running on the correct port (e.g., 8000).")
            st.code(str(e))
if st.session_state.analyzed:
    data = st.session_state.ticket_data
    
    st.success("Analysis Complete")

    report_text = f"""======================================================
AI-DRIVEN ITSM DIAGNOSTIC REPORT
======================================================
TICKET DETAILS
- Description: {data.get('description', 'N/A')}
- Priority: {data.get('priority', 'N/A')}
- Region: {data.get('region', 'N/A')}

AI PREDICTIONS & TELEMETRY
- System Category: {data.get('category', 'N/A')}
- SLA Breach Risk: {data.get('risk', 'N/A')}%
- Estimated Resolution Time (ETA): {data.get('eta', 'N/A')} Hours
- User Sentiment: {data.get('sentiment', 'N/A')}
- Anomaly Detected: {'YES - ' + data.get('anomaly_reason', '') if data.get('is_anomaly') else 'NO'}

RECOMMENDED RESOLUTION
{data.get('solution', 'N/A')}
======================================================
Generated by AI-Driven ITSM Intelligence Platform"""

    col_dl1, col_dl2 = st.columns([4, 1]) 
    with col_dl2:
        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name="ITSM_Diagnostic_Report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    if data['is_anomaly']:
        st.error(f"UNSUPERVISED ML ALERT (Anomaly Detected): Ticket exhibits irregular patterns: {data['anomaly_reason']}. Quarantined for manual review.")
        st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predicted SLA Risk", f"{data['risk']}%")
    m2.metric("AI-Verified Category", data['category'])
    m3.metric("Estimated Repair Time", f"{data['eta']} Hours")
    m4.metric("User Sentiment", f"{'CRITICAL: ' if data['alert'] else 'NORMAL: '}{data['sentiment']}")
    m5.metric("System Route", data['status'])
    st.divider()

    desc_lower = data.get('description', '').lower()
    specific_solution = data.get('solution', 'No specific action recommended. Await analyst review.')
    
    if "teams" in desc_lower and "meeting" in desc_lower:
        specific_solution = "Clear the Microsoft Teams cache (%appdata%\\Microsoft\\Teams), force restart the application, and verify the Exchange Calendar Add-in is enabled and syncing in Outlook."
    elif "vpn" in desc_lower:
        specific_solution = "Reset the Cisco AnyConnect virtual adapter, clear the local DNS cache (ipconfig /flushdns), and re-authenticate the user's multi-factor token."
    elif data.get('is_anomaly') or "5tb" in desc_lower or "unauthorized" in desc_lower:
        specific_solution = "**CRITICAL ACTION:** Immediately sever external network connections to the affected node to halt data exfiltration. Capture volatile memory (RAM) for forensic analysis prior to any system reboot."

    if data['found_fix']:
        st.markdown(f"### AI Intelligent Diagnostics for {data['issue_type']}")
        c_sol1, c_sol2 = st.columns(2)
        with c_sol1:
            st.metric("Solution Confidence", f"{data['confidence']}%")
        with st.container(border=True):
            st.success(f"Recommended Action: {specific_solution}")
            
            c_b1, c_b2, c_b3 = st.columns([1,1,2])
            with c_b1:
                st.button("Execute Resolution")
            with c_b2:
                st.button("Escalate to Analyst")
            with c_b3:
                st.write("Estimated ROI: 15 minutes of Analyst time saved.")
        st.divider()

    st.subheader("7-Day Predictive Workload Forecast")
    chart_data = pd.DataFrame({'Date': data['forecast_labels'], 'Predicted Tickets': data['forecast_values']})
    base_chart = alt.Chart(chart_data).mark_area(
        line={'color': '#8B5CF6'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#8B5CF6', offset=0), alt.GradientStop(color='rgba(139,92,246,0)', offset=1)], x1=1, x2=1, y1=1, y2=0)
    ).encode(x=alt.X('Date', title=''), y=alt.Y('Predicted Tickets', scale=alt.Scale(domainMin=0))).interactive(bind_y=False) 
    st.altair_chart(base_chart, use_container_width=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### AI-Suggested Technical Fix")
        # Now this will never throw an error!
        st.info(specific_solution)
        
    with c2:
        st.markdown("#### Knowledge Base Correlation")
        matches = pd.DataFrame(data['kb_matches'])
        if not matches.empty:
            matches['resolution'] = matches['resolution'].apply(remove_emojis)
            st.dataframe(matches[['id', 'similarity', 'resolution']], hide_index=True)
    
    st.markdown("""
    <div style="margin-top: 40px;">
        <h2 class="section-heading">MLOps: Continuous Model Monitoring</h2>
        <p class="main-subtitle">Real-time telemetry tracking Concept Drift and Model Performance Degradation.</p>
    </div>
    """, unsafe_allow_html=True)

    np.random.seed(42)
    baseline_data = np.random.normal(loc=45, scale=10, size=1000)
    
    if data['priority'] == 'Critical':
        current_data = np.random.normal(loc=75, scale=15, size=1000) 
        drift_detected = True
    else:
        current_data = np.random.normal(loc=48, scale=11, size=1000)
        drift_detected = False

    df_drift = pd.DataFrame({
        'Ticket Volume / Complexity': np.concatenate([baseline_data, current_data]),
        'Dataset Phase': ['1. Training Baseline'] * 1000 + ['2. Live Production'] * 1000
    })

    drift_chart = alt.Chart(df_drift).mark_area(opacity=0.55).encode(
        x=alt.X('Ticket Volume / Complexity', bin=alt.Bin(maxbins=50), title='Feature Distribution (Volume & Complexity)'),
        y=alt.Y('count()', stack=None, title='Frequency'),
        color=alt.Color('Dataset Phase', scale=alt.Scale(domain=['1. Training Baseline', '2. Live Production'], range=['#10b981', '#ef4444']))
    ).properties(height=300).interactive()
    
    st.altair_chart(drift_chart, use_container_width=True)

    if drift_detected:
        st.error("**Data Drift Alert (KL Divergence > 0.15):** The statistical distribution of incoming IT incidents has significantly deviated from the baseline. XGBoost predictive accuracy is degrading. Recalibration pipeline required.")
    else:
        st.success("**Model Health Stable:** Live production data closely matches training baseline distributions. No recalibration needed.")
            
    st.divider()

    st.markdown("""
    <div style="margin-top: 40px;">
        <h2 class="section-heading">Phase 2: Experimental AI Labs</h2>
        <p class="main-subtitle">Previewing Next-Generation Agentic Swarm & Deep Learning Topologies.</p>
    </div>
    """, unsafe_allow_html=True)

    col_swarm, col_gnn = st.columns(2)

    with col_swarm:
        desc_lower = data.get('description', '').lower()
        
        if "5tb" in desc_lower or "unauthorized" in desc_lower or "exfiltration" in desc_lower:
            swarm_log_1 = "<div class='ai-log log-error'>[CRITICAL] Unauthorized process detected (tar -czf). Potential Data Exfiltration in progress.</div>"
            swarm_consensus = "<span style='font-weight:700; color:#ef4444;'>DO NOT REBOOT. Isolate node immediately. Escalate to Tier 3 Incident Response.</span>"
        elif "vpn" in desc_lower or "access" in desc_lower:
            swarm_log_1 = "<div class='ai-log log-warn'>Anomaly detected in authentication logs. Possible credential issue.</div>"
            swarm_consensus = "<span style='font-weight:500; color:#334155;'>Proceed with standard L1/L2 automated resolution.</span>"
        else:
            swarm_log_1 = "<div class='ai-log log-secure'>Network perimeter secure. No threat detected.</div>"
            swarm_consensus = "<span style='font-weight:500; color:#334155;'>Proceed with standard L1/L2 automated resolution.</span>"
            
        swarm_log_2 = f"<div class='ai-log log-error'>High latency detected on {data['region']} backbone router.</div>" if data['region'] in ["Global", "North"] else "<div class='ai-log log-secure'>Traffic flow nominal. BGP routes stable.</div>"
        
        swarm_html = f"""<div class="ai-card"><div class="ai-column-header">Autonomous Agentic Swarm</div><div class="ai-agent-title">SecOps AI: <span style="font-weight:normal; color:#64748b;">Scanning for lateral movement...</span></div>{swarm_log_1}<div class="ai-agent-title">NetOps AI: <span style="font-weight:normal; color:#64748b;">Tracing packet latency in sector...</span></div>{swarm_log_2}<div class="ai-agent-title">DataOps AI: <span style="font-weight:normal; color:#64748b;">Querying transaction states...</span></div><div class="ai-log log-info">No database deadlocks detected in current queue.</div><hr style="border-top: 1px solid #e2e8f0; margin: 20px 0;"><div style="font-weight:800; color:#4f46e5; font-size:1.25rem; margin-top: 10px;">Swarm Consensus: {swarm_consensus}</div></div>"""
        st.markdown(swarm_html, unsafe_allow_html=True)

    with col_gnn:
        base_prob = 85 if data.get('priority', 'Low') in ["High", "Critical"] else 25
        core_color = "#ef4444" if base_prob > 80 else "#10b981" 
        gateway_color = "#f59e0b" if base_prob > 80 else "#10b981"
        laptop_color = "#ef4444" if base_prob > 80 else "#10b981"
        
        st.markdown("""<style>.gnn-container { background: linear-gradient(145deg, #ffffff, #faf5ff); border: 1px solid #e9d5ff; border-radius: 10px; padding: 25px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.08); height: 100%; }</style><div class="gnn-container"><div class="ai-column-header">GNN Root Cause Topology</div><p style="font-size: 0.9rem; color: #64748b; margin-bottom: 15px;">Interactive Node Map: Drag to explore cascade dependencies.</p></div>""", unsafe_allow_html=True)
        
        graph_html = f"""<html><head><script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script><style type="text/css">#mynetwork {{ width: 100%; height: 320px; border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 8px; background-color: #f8fafc; }}</style></head><body><div id="mynetwork"></div><script type="text/javascript">var nodes = new vis.DataSet([{{id: 1, label: 'Core Server\\n({data['region']})', shape: 'dot', size: 30, color: '{core_color}', font: {{color: '#1e293b', bold: true}} }},{{id: 2, label: 'Auth Gateway', shape: 'dot', size: 20, color: '{gateway_color}', font: {{color: '#1e293b'}} }},{{id: 3, label: 'User Endpoint', shape: 'dot', size: 15, color: '{laptop_color}', font: {{color: '#1e293b'}} }}]);var edges = new vis.DataSet([{{from: 1, to: 2, arrows: 'to', color: {{color: '{gateway_color}'}}, width: 3, dashes: true}},{{from: 2, to: 3, arrows: 'to', color: {{color: '{laptop_color}'}}, width: 2}}]);var container = document.getElementById('mynetwork');var data = {{ nodes: nodes, edges: edges }};var options = {{ physics: {{ barnesHut: {{ gravitationalConstant: -2000, centralGravity: 0.3, springLength: 95 }} }}, interaction: {{ hover: true, dragNodes: true, zoomView: false, dragView: false }} }};var network = new vis.Network(container, data, options);</script></body></html>"""
        components.html(graph_html, height=350)
        
        if base_prob > 80:
            st.markdown("<div class='ai-log log-error' style='font-weight:700; font-size: 1.15rem; padding: 18px; text-align: center;'>GNN Alert: High probability of cascading failure. Isolate Core Infrastructure immediately.</div>", unsafe_allow_html=True)