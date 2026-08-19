"""OA ClinicAI - Osteoarthritis AI Diagnosis dashboard.

Run with:  python -m streamlit run app_dashboard.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

import vision

load_dotenv()

st.set_page_config(page_title="OA ClinicAI", page_icon="🦴", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f4f7fb; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right:1px solid #e5eaf1; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
    #MainMenu, footer { visibility: hidden; }

    /* ---------- cards ---------- */
    .card {
        background:#fff; border:1px solid #e5eaf1; border-radius:14px;
        padding:16px 18px; margin-bottom:16px;
        box-shadow:0 1px 3px rgba(16,40,80,.05);
    }
    .card h4 {
        margin:0 0 14px 0; font-size:.98rem; font-weight:700; color:#0f2942;
        display:flex; align-items:center; gap:8px;
    }

    /* ---------- header ---------- */
    .page-title { font-size:2.4rem; font-weight:800; color:#0f2942; margin:0; line-height:1.15; }
    .page-sub   { color:#5b7189; margin-top:4px; font-size:.98rem; }
    .hero-badge {
        background:#e8f1fe; border:1px solid #cfe1fb; border-radius:12px;
        padding:14px 18px; display:flex; align-items:center; gap:12px;
        color:#1d4ed8; font-weight:600; font-size:.92rem; line-height:1.3;
    }
    .hero-badge .ico { font-size:1.4rem; }
    .guest-chip {
        display:inline-flex; align-items:center; gap:8px; float:right;
        background:#fff; border:1px solid #e5eaf1; border-radius:999px;
        padding:6px 14px; color:#475569; font-size:.86rem; font-weight:600;
    }

    /* ---------- sidebar ---------- */
    .brand { display:flex; align-items:center; gap:11px; margin-bottom:4px; }
    .brand-logo {
        width:42px; height:42px; border-radius:50%;
        background:linear-gradient(135deg,#0e7490,#0891b2);
        display:flex; align-items:center; justify-content:center;
        font-size:1.3rem; flex-shrink:0;
    }
    .brand-name { font-size:1.22rem; font-weight:800; color:#0f2942; line-height:1.1; }
    .brand-tag  { font-size:.76rem; color:#7b8da0; line-height:1.25; }
    .side-quote {
        font-style:italic; color:#8fa3b8; font-size:.92rem;
        line-height:1.5; padding-top:8px;
    }
    /* nav buttons */
    section[data-testid="stSidebar"] .stButton>button {
        width:100%; text-align:left; justify-content:flex-start;
        border:none; background:transparent; color:#475569;
        font-weight:600; font-size:.95rem; padding:.5rem .75rem; border-radius:9px;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background:#f1f5f9; color:#0f2942;
    }
    section[data-testid="stSidebar"] .stButton>button[kind="primary"] {
        background:#e8f1fe; color:#1d4ed8;
    }

    /* ---------- rows / pills / bars ---------- */
    .row { display:flex; justify-content:space-between; align-items:center;
           padding:8px 0; border-bottom:1px solid #f0f3f7; font-size:.9rem; }
    .row:last-child { border-bottom:none; }
    .row .k { color:#5b7189; }
    .row .v { font-weight:600; color:#0f2942; }

    .pill { padding:4px 13px; border-radius:999px; font-size:.82rem; font-weight:600; }
    .pill-red{background:#fee2e2;color:#b91c1c}   .pill-orange{background:#ffedd5;color:#c2410c}
    .pill-yellow{background:#fef3c7;color:#a16207} .pill-green{background:#dcfce7;color:#15803d}
    .pill-blue{background:#dbeafe;color:#1d4ed8}   .pill-grey{background:#eef2f7;color:#475569}

    .banner-red{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
        padding:15px;text-align:center;font-size:1.12rem;font-weight:700;color:#b91c1c}
    .banner-green{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
        padding:15px;text-align:center;font-size:1.12rem;font-weight:700;color:#15803d}

    .bar-bg{background:#eef2f7;border-radius:999px;height:9px;width:100%}
    .bar-fill{height:9px;border-radius:999px}

    /* risk rows: name | bar | level */
    .risk { display:grid; grid-template-columns:1.15fr 1.35fr .6fr;
            align-items:center; gap:10px; padding:9px 0; }
    .risk .n { color:#5b7189; font-size:.88rem; }
    .risk .l { text-align:right; font-weight:600; font-size:.86rem; }

    /* KL strip */
    .kl-wrap { display:flex; gap:7px; }
    .kl { flex:1; text-align:center; border-radius:9px; padding:6px 3px; }
    .kl .thumb { height:52px; border-radius:6px; background:linear-gradient(160deg,#e2e8f0,#cbd5e1);
                 margin-bottom:6px; display:flex; align-items:center; justify-content:center;
                 color:#94a3b8; font-size:1.2rem; }
    .kl .lbl { font-size:.75rem; font-weight:600; }

    /* joint space sub-cards */
    .sub { background:#f8fafc; border:1px solid #eef2f7; border-radius:10px;
           padding:12px 14px; height:100%; }
    .sub .t { color:#5b7189; font-size:.83rem; margin-bottom:6px; }
    .metric-big { font-size:1.75rem; font-weight:800; color:#0f2942; line-height:1.1; }
    .metric-cap { color:#94a3b8; font-size:.78rem; }

    /* recommendations checklist */
    .checklist { background:#f0fdf4; border:1px solid #dcfce7; border-radius:10px; padding:14px 16px; }
    .check { display:flex; gap:10px; align-items:flex-start; padding:6px 0;
             color:#166534; font-size:.9rem; line-height:1.45; }
    .check .tick { color:#22c55e; font-weight:800; flex-shrink:0; }

    .sim-note{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
        border-radius:8px;padding:11px 15px;color:#92400e;font-size:.86rem;margin-bottom:15px}
    .info-note{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
        padding:12px 14px;color:#1e40af;font-size:.87rem}
    .footer-bar{color:#7b8da0;font-size:.85rem;padding-top:10px;
        border-top:1px solid #e5eaf1;margin-top:14px;display:flex;justify-content:space-between}
</style>
""", unsafe_allow_html=True)


def card_open(title, icon=""):
    st.markdown(f"<div class='card'><h4>{icon} {title}</h4>", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def row(key, value_html):
    st.markdown(f"<div class='row'><span class='k'>{key}</span>"
                f"<span class='v'>{value_html}</span></div>", unsafe_allow_html=True)


PILL = {"High": "pill-red", "Moderate": "pill-orange", "Low": "pill-green",
        "None": "pill-grey", "Mild": "pill-yellow", "Marked": "pill-red",
        "Severe": "pill-red", "Doubtful": "pill-yellow"}
BAR = {"High": "#ef4444", "Moderate": "#f59e0b", "Low": "#22c55e"}

NAV = [("Home", "🏠"), ("Prediction", "🧠"), ("Patient Records", "👤"),
       ("Statistics", "📊"), ("About", "ℹ️")]

if "page" not in st.session_state:
    st.session_state.page = "Home"

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='brand-logo'>🦴</div>"
        "<div><div class='brand-name'>OA ClinicAI</div>"
        "<div class='brand-tag'>Smarter Insights<br>for Healthier Joints</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    for name, icon in NAV:
        active = st.session_state.page == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}",
                     type="primary" if active else "secondary",
                     use_container_width=True):
            st.session_state.page = name
            st.rerun()

    st.divider()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

    mode = st.selectbox(
        "Retrieval mode", ["demo", "accurate", "fast"], index=0,
        help="accurate = 87.6% precision (~14s) · demo = 84.4% (~6s) · fast = 82.8% (~4s)",
    )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='side-quote'>“Better Joints<br>for a More<br>Active Tomorrow”</div>",
                unsafe_allow_html=True)

page = st.session_state.page

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("<div class='guest-chip'>👤 Guest</div>", unsafe_allow_html=True)
h1, h2 = st.columns([2.6, 1])
with h1:
    st.markdown("<div class='page-title'>Osteoarthritis AI Diagnosis</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Upload a knee X-ray and get an AI-assisted "
                "assessment with guideline-grounded clinical insights.</div>",
                unsafe_allow_html=True)
with h2:
    st.markdown("<div class='hero-badge'><span class='ico'>🩺</span>"
                "<span>AI Support for Better<br>Clinical Decisions</span></div>",
                unsafe_allow_html=True)
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ==========================================================================
# HOME
# ==========================================================================
if page == "Home":

    if not vision.MODEL_AVAILABLE:
        st.markdown(
            "<div class='sim-note'><b>Demo mode — no vision model connected.</b> "
            "The X-ray prediction, Kellgren–Lawrence grade and joint-space figures are "
            "placeholder values for interface demonstration only. They are <b>not</b> "
            "measurements. The clinical recommendations panel is real — it retrieves "
            "NICE and ACR/AF guideline text.</div>",
            unsafe_allow_html=True,
        )

    top = st.columns([1.05, 1, 1.12, 1.3])

    with top[0]:
        card_open("1. Upload Knee X-ray", "🩻")
        st.caption("Upload a knee X-ray image (JPG, PNG)")
        uploaded = st.file_uploader("X-ray", type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        card_close()

    image_bytes = uploaded.getvalue() if uploaded else None
    result = vision.analyze_xray(image_bytes) if image_bytes else None

    with top[1]:
        card_open("2. Image Preview", "🖼️")
        if uploaded:
            st.image(uploaded, use_container_width=True)
        else:
            st.markdown("<div style='height:175px;display:flex;align-items:center;"
                        "justify-content:center;background:#f8fafc;border:1px dashed #cbd5e1;"
                        "border-radius:10px;color:#94a3b8;font-size:.85rem;'>"
                        "No image uploaded</div>", unsafe_allow_html=True)
        card_close()

    with top[2]:
        card_open("3. AI Prediction Result", "⚕️")
        if result:
            cls = "banner-red" if result["detected"] else "banner-green"
            txt = "Osteoarthritis Detected" if result["detected"] else "No Osteoarthritis Detected"
            st.markdown(f"<div class='{cls}'>{txt}</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            conf = result["confidence"]
            st.markdown(
                f"<div class='row' style='border:none;padding-bottom:4px;'>"
                f"<span class='k'>Confidence Score</span><span class='v'>{conf}%</span></div>"
                f"<div class='bar-bg'><div class='bar-fill' style='width:{conf}%;"
                f"background:linear-gradient(90deg,#86efac,#22c55e);'></div></div>",
                unsafe_allow_html=True)
            sev = result["severity"]
            row("Severity Level", f"<span class='pill {PILL.get(sev,'pill-grey')}'>{sev}</span>")
            row("Kellgren–Lawrence Grade",
                f"<span class='pill pill-orange'>Grade {result['kl_grade']}</span>")
        else:
            st.info("Upload an X-ray to see a prediction.")
        card_close()

    with top[3]:
        card_open("Kellgren–Lawrence Grading", "🦿")
        g = result["kl_grade"] if result else None
        chips = "".join(
            f"<div class='kl' style=\"border:{'2px solid #2563eb' if i==g else '1px solid #e5eaf1'};"
            f"background:{'#eff6ff' if i==g else '#fff'};\">"
            f"<div class='thumb'>🦴</div>"
            f"<div class='lbl' style=\"color:{'#1d4ed8' if i==g else '#64748b'};\">Grade {i}</div>"
            f"</div>" for i in range(5))
        st.markdown(f"<div class='kl-wrap'>{chips}</div>", unsafe_allow_html=True)
        if result:
            st.markdown(f"<div class='info-note' style='margin-top:12px;'>"
                        f"<b>Grade {g}:</b> {result['kl_description']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='info-note' style='margin-top:12px;'>"
                        "Upload an X-ray to highlight the predicted grade.</div>",
                        unsafe_allow_html=True)
        card_close()

    # ---------------- Row 2 ----------------
    mid = st.columns([1.15, 1, 1.15])

    with mid[0]:
        card_open("Patient Information", "👤")
        c1, c2 = st.columns([2, 1])
        age = c1.number_input("Age", 18, 100, 62)
        c2.markdown("<div style='padding-top:34px;color:#7b8da0;font-size:.85rem'>years</div>",
                    unsafe_allow_html=True)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        c3, c4 = st.columns([2, 1])
        bmi = c3.number_input("BMI", 12.0, 60.0, 28.4, step=0.1)
        c4.markdown("<div style='padding-top:34px;color:#7b8da0;font-size:.85rem'>kg/m²</div>",
                    unsafe_allow_html=True)
        knee = st.selectbox("Affected Knee", ["Right Knee", "Left Knee", "Both"])
        c5, c6 = st.columns([2, 1])
        duration = c5.number_input("Symptoms Duration", 0, 600, 12)
        c6.markdown("<div style='padding-top:34px;color:#7b8da0;font-size:.85rem'>months</div>",
                    unsafe_allow_html=True)
        prev_injury = st.checkbox("Previous joint injury")
        family_hx = st.checkbox("Family history of OA")
        heavy_load = st.checkbox("High physical/occupational load")
        card_close()

    with mid[1]:
        card_open("Risk Factors", "⚠️")
        for rf in vision.risk_factors(age, bmi, prev_injury, family_hx, heavy_load):
            c = BAR.get(rf["level"], "#94a3b8")
            st.markdown(
                f"<div class='risk'><div class='n'>{rf['name']}</div>"
                f"<div class='bar-bg'><div class='bar-fill' style='width:{rf['value']*100:.0f}%;"
                f"background:{c};'></div></div>"
                f"<div class='l' style='color:{c};'>{rf['level']}</div></div>",
                unsafe_allow_html=True)
        st.caption("Rule-based thresholds from established OA risk factors.")
        card_close()

    with mid[2]:
        card_open("Joint Space Analysis", "📐")
        if result:
            a, b = st.columns(2)
            a.markdown(f"<div class='sub'><div class='t'>Joint Space Width</div>"
                       f"<div class='metric-big'>{result['joint_space_mm']} mm</div>"
                       f"<div class='metric-cap'>Normal: {result['joint_space_normal']}</div></div>",
                       unsafe_allow_html=True)
            osteo = result["osteophytes"]
            opill = "pill-red" if osteo == "Detected" else "pill-green"
            b.markdown(f"<div class='sub'><div class='t'>Osteophyte Detection</div>"
                       f"<span class='pill {opill}'>{osteo}</span></div>",
                       unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            c, d = st.columns(2)
            c.markdown(f"<div class='sub'><div class='t'>Bone Sclerosis</div>"
                       f"<span class='pill {PILL.get(result['sclerosis'],'pill-grey')}'>"
                       f"{result['sclerosis']}</span></div>", unsafe_allow_html=True)
            d.markdown(f"<div class='sub'><div class='t'>Joint Alignment</div>"
                       f"<span class='pill pill-blue'>{result['alignment']}</span></div>",
                       unsafe_allow_html=True)
        else:
            st.info("Upload an X-ray to see joint space analysis.")
        card_close()

    # ---------------- Row 3 ----------------
    bot = st.columns([1.5, 1, 1])

    with bot[0]:
        card_open("Clinical Recommendations", "📋")
        if st.button("⚡ Generate guideline-cited recommendations",
                     type="primary", use_container_width=True):
            if not api_key:
                st.error("Enter your Gemini API key in the sidebar first.")
            else:
                q = (f"What are the recommended management options for a {age}-year-old "
                     f"patient with knee osteoarthritis, BMI {bmi}, symptoms for "
                     f"{duration} months?")
                with st.spinner("Retrieving guideline evidence..."):
                    try:
                        from backend import generate_response
                        ans, src = generate_response(q, api_key, top_n=5, mode=mode)
                        st.session_state["answer"] = ans
                        st.session_state["sources"] = src
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        if st.session_state.get("answer"):
            st.markdown(st.session_state["answer"])
            if st.session_state.get("sources"):
                st.markdown("**Retrieved evidence**")
                for i, s in enumerate(st.session_state["sources"], 1):
                    with st.expander(f"[{i}] {s['doc_name']} · p.{s['page']} · "
                                     f"confidence {s['confidence']:.0f}%"):
                        st.write(s["text"])
                        st.markdown(f"[Source link]({s['url']})")
        else:
            items = [
                "Consider non-pharmacological management (exercise, weight control).",
                "NSAIDs for pain management if needed.",
                "Physiotherapy to improve joint function.",
                "Regular follow-up and monitor progression.",
                "Consider orthopaedic consultation if symptoms worsen.",
            ]
            st.markdown("<div class='checklist'>" + "".join(
                f"<div class='check'><span class='tick'>✓</span><span>{t}</span></div>"
                for t in items) + "</div>", unsafe_allow_html=True)
            st.caption("General guidance only — click above for recommendations "
                       "retrieved and cited from NICE / ACR guidelines.")
        card_close()

    with bot[1]:
        card_open("Additional Notes", "💬")
        st.markdown("<div class='info-note'>This AI tool is intended to support clinical "
                    "decision-making and should not replace professional medical judgment."
                    "</div>", unsafe_allow_html=True)
        st.text_area("Notes", placeholder="Add clinician notes...",
                     label_visibility="collapsed", height=110)
        card_close()

    with bot[2]:
        card_open("Export Results", "📥")
        summary = "No analysis yet."
        if result:
            summary = (
                "OA ClinicAI - assessment summary\n"
                "SIMULATED OUTPUT - no vision model connected.\n\n"
                f"Patient: {age}y {gender}, BMI {bmi}, {knee}, symptoms {duration} months\n"
                f"Finding: {'OA detected' if result['detected'] else 'No OA detected'}\n"
                f"KL grade: {result['kl_grade']} ({result['severity']})\n"
                f"Confidence: {result['confidence']}%\n"
                f"Joint space: {result['joint_space_mm']} mm\n"
                f"Osteophytes: {result['osteophytes']}\n"
                f"Sclerosis: {result['sclerosis']}\n"
                f"Alignment: {result['alignment']}\n")
        st.download_button("📄  Download Report (TXT)", summary,
                           file_name="oa_clinicai_report.txt",
                           type="primary", use_container_width=True)
        st.button("💾  Save to Patient Records", use_container_width=True,
                  disabled=True, help="Not implemented yet")
        if st.button("🔄  New Prediction", use_container_width=True):
            st.session_state.pop("answer", None)
            st.session_state.pop("sources", None)
            st.rerun()
        card_close()

# ==========================================================================
# PREDICTION  (evidence Q&A - fully working)
# ==========================================================================
elif page == "Prediction":
    st.markdown("### 🔎 Ask a clinical question")
    st.caption("Answers are grounded in retrieved NICE NG226 and ACR/AF 2019 guideline text.")

    query = st.text_area("Question", label_visibility="collapsed",
                         placeholder="Example: What is the clinical role of duloxetine "
                                     "in managing knee osteoarthritis?")

    examples = ["Are topical NSAIDs recommended before oral NSAIDs?",
                "What type of exercise is recommended for knee osteoarthritis?",
                "When should a patient be referred for joint replacement surgery?"]
    for col, ex in zip(st.columns(3), examples):
        if col.button(ex, use_container_width=True):
            query = ex

    if st.button("⚡ Retrieve evidence & answer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a question first.")
        elif not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("Retrieving evidence..."):
                try:
                    from backend import generate_response
                    ans, src = generate_response(query, api_key, top_n=5, mode=mode)
                    st.markdown("### Response")
                    st.markdown(ans)
                    if src:
                        st.markdown("#### 📚 Retrieved evidence")
                        for i, s in enumerate(src, 1):
                            with st.expander(f"[{i}] {s['doc_name']} · p.{s['page']} · "
                                             f"confidence {s['confidence']:.0f}%"):
                                st.write(s["text"])
                                st.markdown(f"[Source link]({s['url']})")
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ==========================================================================
# PLACEHOLDER PAGES
# ==========================================================================
elif page in ("Patient Records", "Statistics"):
    st.markdown(f"### {page}")
    st.info(f"**{page} is not implemented yet.** This page is part of the interface "
            "layout; no data store or analytics is connected behind it.")

# ==========================================================================
# ABOUT
# ==========================================================================
else:
    st.markdown("### About OA ClinicAI")
    st.markdown("""
**Evidence retrieval (real).** Hybrid search over NICE NG226 and ACR/AF 2019
osteoarthritis guidelines: PubMedBERT embeddings + BM25 keyword search, fused with
reciprocal rank fusion, reranked by the MedCPT biomedical cross-encoder, and answered
by Gemini strictly over the retrieved text.

| Configuration | Precision@5 | Confidence |
|---|---|---|
| Previous (ms-marco-MiniLM-L-6) | 78.0% | 71.0% |
| Current (MedCPT + fixes) | **87.6%** | **88.2%** |

Confidence is Platt-calibrated so the reported figure tracks measured precision
rather than a raw model score.
""")
    if not vision.MODEL_AVAILABLE:
        st.warning("**X-ray analysis is not implemented.** No trained vision model is "
                   "connected. The grading and joint-space values on the Home page are "
                   "placeholders for interface demonstration only.")

st.markdown("<div class='footer-bar'><span>OA ClinicAI · Built with Streamlit</span>"
            "<span><i>Improving lives through smarter musculoskeletal care</i></span></div>",
            unsafe_allow_html=True)
