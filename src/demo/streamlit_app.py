"""
NEXUS Streamlit Demo Application

Interactive demo for the NEXUS Maternal-Neonatal Care Platform.
Built with Google HAI-DEF models for the MedGemma Impact Challenge.

HAI-DEF Models Used:
- MedSigLIP: Medical image analysis (anemia, jaundice detection)
- HeAR: Health acoustic representations (cry analysis)
- MedGemma: Clinical reasoning and synthesis
"""

import streamlit as st
from pathlib import Path
import sys
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="NEXUS - Maternal-Neonatal Care",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_anemia_detector():
    """Load anemia detector model."""
    from nexus.anemia_detector import AnemiaDetector
    return AnemiaDetector()


@st.cache_resource
def load_jaundice_detector():
    """Load jaundice detector model."""
    from nexus.jaundice_detector import JaundiceDetector
    return JaundiceDetector()


@st.cache_resource
def load_cry_analyzer():
    """Load cry analyzer."""
    from nexus.cry_analyzer import CryAnalyzer
    return CryAnalyzer()


@st.cache_resource
def load_clinical_synthesizer():
    """Load clinical synthesizer (MedGemma)."""
    import os
    from nexus.clinical_synthesizer import ClinicalSynthesizer
    use_medgemma = os.environ.get("NEXUS_USE_MEDGEMMA", "true").lower() != "false"
    return ClinicalSynthesizer(use_medgemma=use_medgemma)


def get_hai_def_info():
    """Get HAI-DEF models information."""
    return {
        "MedSigLIP": {
            "name": "MedSigLIP (google/medsiglip-448)",
            "use": "Image analysis for anemia and jaundice detection",
            "method": "Zero-shot classification with medical prompts",
            "accuracy": "80-98% expected"
        },
        "HeAR": {
            "name": "HeAR (google/hear-pytorch)",
            "use": "Infant cry analysis for asphyxia detection",
            "method": "Health acoustic embeddings + classification",
            "accuracy": "85-93% expected"
        },
        "MedGemma": {
            "name": "MedGemma 4B (google/medgemma-4b-it)",
            "use": "Clinical reasoning and recommendation synthesis",
            "method": "WHO IMNCI protocol-based synthesis",
            "accuracy": "Rule-based clinical guidelines"
        }
    }


def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">NEXUS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Maternal-Neonatal Care Platform</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## 🏥 NEXUS")
        st.markdown("---")

        assessment_type = st.radio(
            "Select Assessment Type",
            [
                "Maternal Anemia Screening",
                "Neonatal Jaundice Detection",
                "Cry Analysis",
                "Combined Assessment",
                "HAI-DEF Models Info"
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### About NEXUS")
        st.markdown("""
        NEXUS uses AI to provide non-invasive screening for:
        - **Maternal Anemia** via conjunctiva imaging
        - **Neonatal Jaundice** via skin color analysis
        - **Birth Asphyxia** via cry pattern analysis

        Built with **Google HAI-DEF models** for the MedGemma Impact Challenge 2026.
        """)

        st.markdown("---")
        st.markdown("### 🤖 HAI-DEF Models")
        st.markdown("""
        - **MedSigLIP**: Vision
        - **HeAR**: Audio
        - **MedGemma**: Clinical AI
        """)

    # Main content based on selection
    if assessment_type == "Maternal Anemia Screening":
        render_anemia_screening()
    elif assessment_type == "Neonatal Jaundice Detection":
        render_jaundice_detection()
    elif assessment_type == "Cry Analysis":
        render_cry_analysis()
    elif assessment_type == "Combined Assessment":
        render_combined_assessment()
    else:
        render_hai_def_info()


def render_anemia_screening():
    """Render anemia screening interface."""
    st.header("Maternal Anemia Screening")
    st.markdown("Upload a clear image of the inner eyelid (conjunctiva) for anemia screening.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a conjunctiva image",
            type=["jpg", "jpeg", "png"],
            key="anemia_upload"
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Analysis Results")

        if uploaded_file:
            with st.spinner("Analyzing image..."):
                try:
                    detector = load_anemia_detector()

                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    result = detector.detect(tmp_path)
                    color_info = detector.analyze_color_features(tmp_path)

                    # Display results
                    risk_class = f"risk-{result['risk_level']}"
                    st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)

                    if result["is_anemic"]:
                        st.error("⚠️ ANEMIA DETECTED")
                    else:
                        st.success("✅ No Anemia Detected")

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col_b:
                        st.metric("Risk Level", result['risk_level'].upper())
                    with col_c:
                        st.metric("Est. Hemoglobin", f"{color_info['estimated_hemoglobin']} g/dL")

                    # Recommendation
                    st.markdown("### Recommendation")
                    st.info(result["recommendation"])

                    # Color analysis
                    with st.expander("Technical Details"):
                        st.json({
                            "anemia_score": round(result["anemia_score"], 3),
                            "healthy_score": round(result["healthy_score"], 3),
                            "red_ratio": round(color_info["red_ratio"], 3),
                            "pallor_index": round(color_info["pallor_index"], 3),
                        })

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")
        else:
            st.info("👆 Upload an image to begin analysis")


def render_jaundice_detection():
    """Render jaundice detection interface."""
    st.header("Neonatal Jaundice Detection")
    st.markdown("Upload an image of the newborn's skin or sclera for jaundice assessment.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a neonatal image",
            type=["jpg", "jpeg", "png"],
            key="jaundice_upload"
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Patient info
        st.subheader("Patient Information (Optional)")
        age_days = st.number_input("Age (days)", min_value=0, max_value=28, value=3)
        birth_weight = st.number_input("Birth weight (grams)", min_value=500, max_value=5000, value=3000)

    with col2:
        st.subheader("Analysis Results")

        if uploaded_file:
            with st.spinner("Analyzing image..."):
                try:
                    detector = load_jaundice_detector()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    result = detector.detect(tmp_path)
                    zone_info = detector.analyze_kramer_zones(tmp_path)

                    # Display results
                    risk_class = "risk-high" if result["needs_phototherapy"] else (
                        "risk-medium" if result["severity"] in ["moderate", "mild"] else "risk-low"
                    )
                    st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)

                    if result["has_jaundice"]:
                        st.warning(f"⚠️ JAUNDICE DETECTED - {result['severity'].upper()}")
                    else:
                        st.success("✅ No Significant Jaundice")

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Est. Bilirubin", f"{result['estimated_bilirubin']} mg/dL")
                    with col_b:
                        st.metric("Severity", result['severity'].upper())
                    with col_c:
                        st.metric("Kramer Zone", zone_info['kramer_zone'])

                    # Phototherapy indicator
                    if result["needs_phototherapy"]:
                        st.error("🔆 PHOTOTHERAPY RECOMMENDED")

                    # Recommendation
                    st.markdown("### Recommendation")
                    st.info(result["recommendation"])

                    # Zone analysis
                    with st.expander("Kramer Zone Analysis"):
                        st.write(f"**Zone**: {zone_info['kramer_zone']} - {zone_info['zone_description']}")
                        st.write(f"**Yellow Index**: {zone_info['yellow_index']}")
                        st.progress(min(zone_info['yellow_index'] * 2, 1.0))

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")
        else:
            st.info("👆 Upload an image to begin analysis")


def render_cry_analysis():
    """Render cry analysis interface."""
    st.header("Infant Cry Analysis")
    st.markdown("Upload an audio recording of the infant's cry for analysis.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose a cry audio file",
            type=["wav", "mp3", "ogg"],
            key="cry_upload"
        )

        if uploaded_file:
            st.audio(uploaded_file)

    with col2:
        st.subheader("Analysis Results")

        if uploaded_file:
            with st.spinner("Analyzing cry..."):
                try:
                    analyzer = load_cry_analyzer()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    result = analyzer.analyze(tmp_path)

                    # Display results
                    risk_class = f"risk-{result['risk_level']}"
                    st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)

                    if result["is_abnormal"]:
                        st.error("⚠️ ABNORMAL CRY PATTERN DETECTED")
                    else:
                        st.success("✅ Normal Cry Pattern")

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Asphyxia Risk", f"{result['asphyxia_risk']:.1%}")
                    with col_b:
                        st.metric("Cry Type", result['cry_type'].title())
                    with col_c:
                        st.metric("F0 (Pitch)", f"{result['features']['f0_mean']:.0f} Hz")

                    # Recommendation
                    st.markdown("### Recommendation")
                    st.info(result["recommendation"])

                    # Acoustic features
                    with st.expander("Acoustic Features"):
                        st.json(result["features"])

                except Exception as e:
                    st.error(f"Error analyzing audio: {e}")
        else:
            st.info("👆 Upload an audio file to begin analysis")


def render_combined_assessment():
    """Render combined assessment interface using Clinical Synthesizer."""
    st.header("Combined Clinical Assessment")
    st.markdown("""
    Upload multiple inputs for a comprehensive assessment using **MedGemma Clinical Synthesizer**.
    This combines findings from all HAI-DEF models to provide integrated clinical recommendations.
    """)

    # Initialize session state for findings
    if "findings" not in st.session_state:
        st.session_state.findings = {
            "anemia": None,
            "jaundice": None,
            "cry": None
        }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🩸 Anemia Screening")
        anemia_file = st.file_uploader(
            "Conjunctiva image",
            type=["jpg", "jpeg", "png"],
            key="combined_anemia"
        )
        if anemia_file:
            st.image(anemia_file, use_container_width=True)
            with st.spinner("Analyzing..."):
                try:
                    detector = load_anemia_detector()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(anemia_file.getvalue())
                        result = detector.detect(tmp.name)
                        st.session_state.findings["anemia"] = result
                        if result["is_anemic"]:
                            st.error(f"Anemia: {result['risk_level'].upper()}")
                        else:
                            st.success("No Anemia")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("👶 Jaundice Detection")
        jaundice_file = st.file_uploader(
            "Neonatal skin image",
            type=["jpg", "jpeg", "png"],
            key="combined_jaundice"
        )
        if jaundice_file:
            st.image(jaundice_file, use_container_width=True)
            with st.spinner("Analyzing..."):
                try:
                    detector = load_jaundice_detector()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(jaundice_file.getvalue())
                        result = detector.detect(tmp.name)
                        st.session_state.findings["jaundice"] = result
                        if result["has_jaundice"]:
                            st.warning(f"Jaundice: {result['severity'].upper()}")
                        else:
                            st.success("No Jaundice")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col3:
        st.subheader("🔊 Cry Analysis")
        cry_file = st.file_uploader(
            "Cry audio",
            type=["wav", "mp3", "ogg"],
            key="combined_cry"
        )
        if cry_file:
            st.audio(cry_file)
            with st.spinner("Analyzing..."):
                try:
                    analyzer = load_cry_analyzer()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(cry_file.getvalue())
                        result = analyzer.analyze(tmp.name)
                        st.session_state.findings["cry"] = result
                        if result["is_abnormal"]:
                            st.error(f"Abnormal Cry: {result['risk_level'].upper()}")
                        else:
                            st.success("Normal Cry")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Clinical Synthesis Section
    st.markdown("---")
    st.subheader("🏥 Clinical Synthesis (MedGemma)")

    # Check if any findings are available
    has_findings = any(v is not None for v in st.session_state.findings.values())

    if has_findings:
        if st.button("Generate Clinical Synthesis", type="primary"):
            with st.spinner("Synthesizing findings with MedGemma..."):
                try:
                    synthesizer = load_clinical_synthesizer()

                    # Prepare findings dict
                    findings = {}
                    if st.session_state.findings["anemia"]:
                        findings["anemia"] = st.session_state.findings["anemia"]
                    if st.session_state.findings["jaundice"]:
                        findings["jaundice"] = st.session_state.findings["jaundice"]
                    if st.session_state.findings["cry"]:
                        findings["cry"] = st.session_state.findings["cry"]

                    synthesis = synthesizer.synthesize(findings)

                    # Display synthesis results
                    severity_level = synthesis.get("severity_level", "GREEN")
                    severity_colors = {
                        "GREEN": ("🟢", "#d4edda", "#155724"),
                        "YELLOW": ("🟡", "#fff3cd", "#856404"),
                        "RED": ("🔴", "#f8d7da", "#721c24")
                    }
                    emoji, bg_color, text_color = severity_colors.get(severity_level, ("⚪", "#f8f9fa", "#000"))

                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: {text_color}; margin: 0;">{emoji} Severity: {severity_level}</h3>
                        <p style="color: {text_color}; font-size: 1.1rem; margin-top: 0.5rem;">{synthesis.get('severity_description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Summary
                    st.markdown("### Summary")
                    st.info(synthesis.get("summary", "No summary available"))

                    # Actions
                    if synthesis.get("immediate_actions"):
                        st.markdown("### Immediate Actions")
                        for action in synthesis["immediate_actions"]:
                            st.markdown(f"- {action}")

                    # Referral
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("### Referral Status")
                        if synthesis.get("referral_needed"):
                            st.error(f"⚠️ REFERRAL NEEDED: {synthesis.get('referral_urgency', 'standard').upper()}")
                        else:
                            st.success("✅ No referral needed")

                    with col_b:
                        st.markdown("### Follow-up")
                        st.info(synthesis.get("follow_up", "Schedule routine follow-up"))

                    # Technical details
                    with st.expander("Technical Details"):
                        st.json({
                            "model": synthesis.get("model"),
                            "generated_at": synthesis.get("generated_at"),
                            "urgent_conditions": synthesis.get("urgent_conditions", []),
                        })

                except Exception as e:
                    st.error(f"Error generating synthesis: {e}")
    else:
        st.info("👆 Upload at least one input (image or audio) to generate clinical synthesis")


def render_hai_def_info():
    """Render HAI-DEF models information."""
    st.header("Google HAI-DEF Models")
    st.markdown("""
    NEXUS is built using **Google Health AI Developer Foundations (HAI-DEF)** models,
    designed specifically for healthcare applications in resource-limited settings.
    """)

    hai_def = get_hai_def_info()

    # MedSigLIP
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🖼️ MedSigLIP")
        st.info("google/medsiglip-448\n\nHAI-DEF Vision Model")
    with col2:
        info = hai_def["MedSigLIP"]
        st.markdown(f"**Model**: {info['name']}")
        st.markdown(f"**Use Case**: {info['use']}")
        st.markdown(f"**Method**: {info['method']}")
        st.markdown(f"**Expected Accuracy**: {info['accuracy']}")
        st.markdown("""
        MedSigLIP enables zero-shot medical image classification using
        text prompts, allowing detection of conditions without requiring
        large labeled training datasets.
        """)

    # HeAR
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🔊 HeAR")
        st.info("google/hear-pytorch\n\nHAI-DEF Audio Model")
    with col2:
        info = hai_def["HeAR"]
        st.markdown(f"**Model**: {info['name']}")
        st.markdown(f"**Use Case**: {info['use']}")
        st.markdown(f"**Method**: {info['method']}")
        st.markdown(f"**Expected Accuracy**: {info['accuracy']}")
        st.markdown("""
        HeAR (Health Acoustic Representations) analyzes audio signals
        for health indicators. In NEXUS, it analyzes infant cry patterns
        to detect potential birth asphyxia.
        """)

    # MedGemma
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🧠 MedGemma")
        st.info("google/medgemma-4b-it\n\nHAI-DEF Language Model")
    with col2:
        info = hai_def["MedGemma"]
        st.markdown(f"**Model**: {info['name']}")
        st.markdown(f"**Use Case**: {info['use']}")
        st.markdown(f"**Method**: {info['method']}")
        st.markdown(f"**Expected Accuracy**: {info['accuracy']}")
        st.markdown("""
        MedGemma provides clinical reasoning capabilities, synthesizing
        multiple findings into actionable recommendations following
        WHO IMNCI protocols for maternal and neonatal care.
        """)

    # Competition Info
    st.markdown("---")
    st.subheader("🏆 MedGemma Impact Challenge 2026")
    st.markdown("""
    NEXUS is being developed for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge-2026)
    on Kaggle.

    **Competition Focus**: Solutions for resource-limited healthcare settings using HAI-DEF models.

    **NEXUS Impact**:
    - 📍 Target: Sub-Saharan Africa and South Asia
    - 👩‍⚕️ Users: Community Health Workers
    - 🎯 Goals: Reduce maternal/neonatal mortality
    - 📱 Deployment: Offline-capable mobile app
    """)


# Footer
def render_footer():
    """Render footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>NEXUS - Built with Google HAI-DEF for MedGemma Impact Challenge 2026</p>
        <p>⚠️ This is a screening tool only. Always confirm with laboratory tests.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    render_footer()
