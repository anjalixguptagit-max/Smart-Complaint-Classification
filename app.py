import streamlit as st
import joblib
import numpy as np
import json
import base64


# =========================
# LOAD MODEL & FILES
# =========================

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

with open("metrics.json", "r") as f:
    metrics = json.load(f)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Smart Complaint AI",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# BACKGROUND IMAGE
# =========================

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


background_image = get_base64_image("assets/background.png")


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    f"""
    <style>

    /* =========================
       MAIN BACKGROUND
       ========================= */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(2, 4, 12, 0.72),
                rgba(3, 5, 15, 0.82)
            ),
            url(
                "data:image/png;base64,{background_image}"
            );

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}


    /* =========================
       BACKGROUND GLOW
       ========================= */

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.18),
                transparent 30%
            ),

            radial-gradient(
                circle at 90% 10%,
                rgba(168, 85, 247, 0.16),
                transparent 30%
            );

        pointer-events: none;
        z-index: 0;
    }}


    /* =========================
       MAIN CONTAINER
       ========================= */

    .block-container {{
        max-width: 1150px;
        padding-top: 35px;
        padding-bottom: 55px;

        position: relative;
        z-index: 1;
    }}


    /* =========================
       HEADINGS
       ========================= */

    h1 {{
        font-size: 43px !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        color: #ffffff !important;

        text-shadow:
            0 3px 25px rgba(0,0,0,0.45);
    }}


    h2,
    h3 {{
        color: #f8fafc !important;
        font-weight: 750 !important;
    }}


    /* =========================
       SUBTITLE
       ========================= */

    .subtitle-text {{
        color: #c1c9d8;
        font-size: 16px;

        margin-top: -8px;
        margin-bottom: 28px;

        text-shadow:
            0 2px 10px rgba(0,0,0,0.5);
    }}


    /* =========================
       DIVIDER
       ========================= */

    hr {{
        border-color:
            rgba(255,255,255,0.15) !important;
    }}


    /* =========================
       METRIC CARDS
       ========================= */

    div[data-testid="stMetric"] {{
        background:
            linear-gradient(
                145deg,
                rgba(15,23,42,0.82),
                rgba(5,10,24,0.76)
            );

        border:
            1px solid rgba(148,163,184,0.20);

        border-radius: 18px;

        padding: 20px;

        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);

        box-shadow:
            0 15px 40px rgba(0,0,0,0.35);
    }}


    div[data-testid="stMetricLabel"] {{
        color: #aab4c6 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }}


    div[data-testid="stMetricValue"] {{
        color: #ffffff !important;
        font-weight: 800 !important;
    }}


    /* =========================
       TEXT AREA
       ========================= */

    textarea {{
        background:
            rgba(5,10,24,0.78) !important;

        color:
            #ffffff !important;

        border:
            1px solid rgba(148,163,184,0.25) !important;

        border-radius:
            15px !important;

        font-size:
            15px !important;

        backdrop-filter:
            blur(12px);

        -webkit-backdrop-filter:
            blur(12px);

        box-shadow:
            0 12px 35px rgba(0,0,0,0.25);
    }}


    textarea:focus {{
        border-color:
            #818cf8 !important;

        box-shadow:
            0 0 0 1px #6366f1,
            0 0 30px rgba(99,102,241,0.22) !important;
    }}


    /* =========================
       BUTTON
       ========================= */

    .stButton > button {{
        width: 100%;
        height: 54px;

        border-radius: 14px;

        border:
            1px solid rgba(196,181,253,0.35);

        background:
            linear-gradient(
                90deg,
                #4f46e5,
                #7c3aed,
                #9333ea
            );

        color: #ffffff;

        font-size: 15px;
        font-weight: 700;

        box-shadow:
            0 10px 30px rgba(79,70,229,0.35);

        transition:
            all 0.2s ease;
    }}


    .stButton > button:hover {{
        background:
            linear-gradient(
                90deg,
                #6366f1,
                #8b5cf6,
                #a855f7
            );

        border-color:
            rgba(255,255,255,0.45);

        box-shadow:
            0 14px 40px rgba(124,58,237,0.45);

        transform:
            translateY(-1px);
    }}


    /* =========================
       RESULT / ALERT CARDS
       ========================= */

    div[data-testid="stAlert"] {{
        border-radius:
            16px !important;

        border-width:
            1px !important;

        backdrop-filter:
            blur(12px);

        -webkit-backdrop-filter:
            blur(12px);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.25);
    }}


    /* =========================
       LABELS
       ========================= */

    label {{
        color:
            #d5dbea !important;

        font-weight:
            600 !important;
    }}


    /* =========================
       FOOTER
       ========================= */

    .footer-text {{
        text-align: center;

        color:
            #9aa5ba;

        font-size:
            12px;

        margin-top:
            32px;

        text-shadow:
            0 2px 8px rgba(0,0,0,0.5);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HEADER
# =========================

st.markdown(
    "# 📩 Smart Complaint Classification"
)

st.markdown(
    '<div class="subtitle-text">'
    'AI-powered complaint classification and intelligent department routing'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================
# MODEL PERFORMANCE
# =========================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Test Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Macro F1",
        f"{metrics['macro_f1']:.2f}"
    )

with col3:
    st.metric(
        "Categories",
        metrics["categories"]
    )

with col4:
    st.metric(
        "Model",
        "Linear SVM"
    )


# =========================
# COMPLAINT INPUT
# =========================

st.divider()

st.subheader("📝 Analyze Customer Complaint")

complaint = st.text_area(
    "Enter your complaint",

    placeholder=(
        "Example: My credit card was charged for a transaction "
        "I did not make..."
    ),

    height=160
)


analyze = st.button(
    "🔍  Analyze Complaint",
    use_container_width=True
)


# =========================
# CLASSIFICATION
# =========================

if analyze:

    if complaint.strip() == "":
        st.warning(
            "Please enter a complaint first."
        )

    else:

        # Convert complaint into TF-IDF
        complaint_tfidf = tfidf.transform(
            [complaint]
        )


        # Predict category
        prediction = model.predict(
            complaint_tfidf
        )[0]


        # =========================
        # CONFIDENCE SCORE
        # =========================

        scores = model.decision_function(
            complaint_tfidf
        )[0]

        exp_scores = np.exp(
            scores - np.max(scores)
        )

        confidence_scores = (
            exp_scores /
            exp_scores.sum()
        )

        predicted_index = list(
            model.classes_
        ).index(prediction)

        confidence = (
            confidence_scores[predicted_index]
            * 100
        )


        # =========================
        # DEPARTMENT MAPPING
        # =========================

        department_map = {

            "credit_card":
                "Credit Card Support",

            "credit_reporting":
                "Credit Reporting Team",

            "debt_collection":
                "Debt Collection Team",

            "mortgages_and_loans":
                "Loans & Mortgage Department",

            "retail_banking":
                "Retail Banking Support"
        }


        department = department_map[
            prediction
        ]


        # =========================
        # RECOMMENDED ACTION
        # =========================

        action_map = {

            "credit_card":
                "Forward the complaint to Credit Card Support for transaction verification.",

            "credit_reporting":
                "Forward the complaint to the Credit Reporting Team for report verification and correction.",

            "debt_collection":
                "Forward the complaint to the Debt Collection Team for collection-related issue review.",

            "mortgages_and_loans":
                "Forward the complaint to the Loans & Mortgage Department for application or loan-status review.",

            "retail_banking":
                "Forward the complaint to Retail Banking Support for transaction and account verification."
        }


        recommended_action = action_map[
            prediction
        ]


        # =========================
        # TOP TF-IDF TERMS
        # =========================

        feature_names = (
            tfidf.get_feature_names_out()
        )

        tfidf_values = (
            complaint_tfidf.toarray()[0]
        )

        top_indices = (
            tfidf_values.argsort()[-5:][::-1]
        )

        top_terms = [

            feature_names[i]

            for i in top_indices

            if tfidf_values[i] > 0
        ]


        # =========================
        # RESULTS
        # =========================

        st.divider()

        st.subheader(
            "🎯 Classification Result"
        )


        r1, r2, r3 = st.columns(3)


        with r1:

            st.info(
                f"**Predicted Category**\n\n"
                f"### {prediction.replace('_', ' ').title()}"
            )


        with r2:

            st.success(
                f"**Confidence Score**\n\n"
                f"### {confidence:.2f}%"
            )


        with r3:

            st.warning(
                f"**Recommended Department**\n\n"
                f"### {department}"
            )


        # =========================
        # RECOMMENDED ACTION
        # =========================

        st.subheader(
            "🤖 Recommended Action"
        )

        st.info(
            recommended_action
        )


        # =========================
        # KEY TERMS
        # =========================

        st.subheader(
            "🔑 Key Terms Detected"
        )

        if top_terms:

            term_string = (
                "  •  ".join(top_terms)
            )

            st.write(
                f"**{term_string}**"
            )

        else:

            st.caption(
                "No significant terms detected."
            )


# =========================
# FOOTER
# =========================

st.divider()

st.markdown(
    '<div class="footer-text">'
    'Smart Complaint Classification'
    ' &nbsp;•&nbsp; '
    'TF-IDF + Linear SVM'
    ' &nbsp;•&nbsp; '
    'Intelligent Complaint Routing'
    '</div>',
    unsafe_allow_html=True
)