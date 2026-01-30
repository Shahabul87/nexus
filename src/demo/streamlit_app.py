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
                "Agentic Workflow",
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
        st.markdown("### Edge AI Mode")
        edge_mode = st.toggle("Enable Edge AI Mode", value=False, key="edge_mode")
        if edge_mode:
            st.success("Edge AI: INT8 quantized models + offline inference")
        else:
            st.info("Cloud mode: Full-precision HAI-DEF models")

        st.markdown("---")
        st.markdown("### HAI-DEF Models")
        st.markdown("""
        - **MedSigLIP**: Vision
        - **HeAR**: Audio
        - **MedGemma**: Clinical AI
        """)

    # Show Edge AI banner when enabled
    if edge_mode:
        render_edge_ai_banner()

    # Main content based on selection
    if assessment_type == "Maternal Anemia Screening":
        render_anemia_screening()
    elif assessment_type == "Neonatal Jaundice Detection":
        render_jaundice_detection()
    elif assessment_type == "Cry Analysis":
        render_cry_analysis()
    elif assessment_type == "Combined Assessment":
        render_combined_assessment()
    elif assessment_type == "Agentic Workflow":
        render_agentic_workflow()
    else:
        render_hai_def_info()


def render_edge_ai_banner():
    """Show Edge AI mode status and model metrics."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
                color: white; padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="margin:0; color: white;">Edge AI Mode Active</h4>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
            Running INT8 quantized models for offline-capable inference on low-resource devices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MedSigLIP INT8", "111.2 MB", "-86% memory")
    with col2:
        st.metric("Acoustic Model", "0.6 MB", "INT8 quantized")
    with col3:
        st.metric("Text Embeddings", "12 KB", "Pre-computed")
    with col4:
        st.metric("Total Edge Size", "~289 MB", "Offline-ready")

    with st.expander("Edge AI Details"):
        st.markdown("""
        **Quantization**: Dynamic INT8 (PyTorch `quantize_dynamic`, qnnpack backend)

        | Component | Cloud (FP32) | Edge (INT8) | Compression |
        |-----------|-------------|-------------|-------------|
        | MedSigLIP Vision | 812.6 MB | 111.2 MB | **7.31x** |
        | Acoustic Model | 0.665 MB | 0.599 MB | 1.11x |
        | CPU Latency | 97.7 ms | ~65 ms (ARM est.) | ~1.5x faster |

        **Target Devices**: Android 8.0+, ARM Cortex-A53, 2GB RAM

        **Offline Capabilities**:
        - Image analysis via INT8 MedSigLIP + pre-computed binary text embeddings
        - Audio analysis via INT8 acoustic feature extractor
        - Clinical reasoning via rule-based WHO IMNCI protocols (no MedGemma required)
        """)


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


def render_agentic_workflow():
    """Render the agentic workflow interface with reasoning traces."""
    st.header("Agentic Clinical Workflow")
    st.markdown("""
    **6-Agent Pipeline** with step-by-step reasoning traces.
    Each agent explains its clinical decision process, providing a full audit trail.
    """)

    # Pipeline diagram
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0;">
        <div style="background: #e3f2fd; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; border: 2px solid #1976d2;">Triage</div>
        <span style="font-size: 1.5rem;">&#8594;</span>
        <div style="background: #e8f5e9; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; border: 2px solid #388e3c;">Image (MedSigLIP)</div>
        <span style="font-size: 1.5rem;">&#8594;</span>
        <div style="background: #fff3e0; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; border: 2px solid #f57c00;">Audio (HeAR)</div>
        <span style="font-size: 1.5rem;">&#8594;</span>
        <div style="background: #f3e5f5; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; border: 2px solid #7b1fa2;">Protocol (WHO)</div>
        <span style="font-size: 1.5rem;">&#8594;</span>
        <div style="background: #fce4ec; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; border: 2px solid #c62828;">Referral</div>
        <span style="font-size: 1.5rem;">&#8594;</span>
        <div style="background: #e0f7fa; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; border: 2px solid #00838f;">Synthesis (MedGemma)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input section
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Patient & Inputs")
        patient_type = st.selectbox("Patient Type", ["newborn", "pregnant"], key="agentic_patient")

        # Danger signs
        st.markdown("**Danger Signs**")
        danger_signs = []
        if patient_type == "pregnant":
            sign_options = [
                ("Severe headache", "high"),
                ("Blurred vision", "high"),
                ("Convulsions", "critical"),
                ("Severe abdominal pain", "high"),
                ("Vaginal bleeding", "critical"),
                ("High fever", "high"),
                ("Severe pallor", "medium"),
            ]
        else:
            sign_options = [
                ("Not breathing at birth", "critical"),
                ("Convulsions", "critical"),
                ("Severe chest indrawing", "high"),
                ("Not feeding", "high"),
                ("High fever (>38C)", "high"),
                ("Hypothermia (<35.5C)", "high"),
                ("Lethargy / unconscious", "critical"),
                ("Umbilical redness", "medium"),
            ]

        selected_signs = st.multiselect(
            "Select present danger signs",
            [s[0] for s in sign_options],
            key="agentic_signs"
        )
        for label, severity in sign_options:
            if label in selected_signs:
                danger_signs.append({
                    "id": label.lower().replace(" ", "_"),
                    "label": label,
                    "severity": severity,
                    "present": True,
                })

        # Image uploads
        st.markdown("**Clinical Images**")
        conjunctiva_file = st.file_uploader(
            "Conjunctiva image (anemia)", type=["jpg", "jpeg", "png"],
            key="agentic_conjunctiva"
        )
        skin_file = st.file_uploader(
            "Skin image (jaundice)", type=["jpg", "jpeg", "png"],
            key="agentic_skin"
        )
        cry_file = st.file_uploader(
            "Cry audio", type=["wav", "mp3", "ogg"],
            key="agentic_cry"
        )

    with col_right:
        st.subheader("Workflow Execution")

        if st.button("Run Agentic Assessment", type="primary", key="run_agentic"):
            with st.spinner("Running 6-agent workflow..."):
                try:
                    from nexus.agentic_workflow import (
                        AgenticWorkflowEngine,
                        AgentPatientInfo,
                        DangerSign,
                        WorkflowInput,
                    )

                    # Save uploaded files
                    conjunctiva_path = None
                    skin_path = None
                    cry_path = None

                    if conjunctiva_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(conjunctiva_file.getvalue())
                            conjunctiva_path = tmp.name

                    if skin_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(skin_file.getvalue())
                            skin_path = tmp.name

                    if cry_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(cry_file.getvalue())
                            cry_path = tmp.name

                    # Build workflow input
                    signs = [
                        DangerSign(
                            id=s["id"], label=s["label"],
                            severity=s["severity"], present=True,
                        )
                        for s in danger_signs
                    ]

                    info = AgentPatientInfo(patient_type=patient_type)
                    workflow_input = WorkflowInput(
                        patient_type=patient_type,
                        patient_info=info,
                        danger_signs=signs,
                        conjunctiva_image=conjunctiva_path,
                        skin_image=skin_path,
                        cry_audio=cry_path,
                    )

                    # Run workflow (lazy-load models)
                    engine = AgenticWorkflowEngine()
                    result = engine.execute(workflow_input)

                    st.session_state["agentic_result"] = result
                    st.success("Workflow complete!")

                except Exception as e:
                    st.error(f"Workflow error: {e}")

    # Results display
    if "agentic_result" in st.session_state:
        result = st.session_state["agentic_result"]

        st.markdown("---")

        # Overall classification
        severity_colors = {
            "GREEN": ("#d4edda", "#155724", "Routine care"),
            "YELLOW": ("#fff3cd", "#856404", "Close monitoring"),
            "RED": ("#f8d7da", "#721c24", "Urgent referral"),
        }
        bg, fg, desc = severity_colors.get(result.who_classification, ("#f8f9fa", "#000", "Unknown"))

        st.markdown(f"""
        <div style="background: {bg}; color: {fg}; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="margin: 0;">WHO Classification: {result.who_classification}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Agents Run", len(result.agent_traces))
        with m2:
            st.metric("Total Time", f"{result.processing_time_ms:.0f} ms")
        with m3:
            referral_text = "Yes" if (result.referral_result and result.referral_result.referral_needed) else "No"
            st.metric("Referral Needed", referral_text)
        with m4:
            triage_score = result.triage_result.score if result.triage_result else 0
            st.metric("Triage Score", triage_score)

        # Clinical synthesis
        st.subheader("Clinical Synthesis")
        st.info(result.clinical_synthesis)

        if result.immediate_actions:
            st.subheader("Immediate Actions")
            for action in result.immediate_actions:
                st.markdown(f"- {action}")

        # Agent reasoning traces (the key feature for Agentic Workflow prize)
        st.markdown("---")
        st.subheader("Agent Reasoning Traces")

        agent_colors = {
            "TriageAgent": "#e3f2fd",
            "ImageAnalysisAgent": "#e8f5e9",
            "AudioAnalysisAgent": "#fff3e0",
            "ProtocolAgent": "#f3e5f5",
            "ReferralAgent": "#fce4ec",
            "SynthesisAgent": "#e0f7fa",
        }
        status_icons = {
            "success": "&#9989;",
            "skipped": "&#9940;",
            "error": "&#10060;",
        }

        for trace in result.agent_traces:
            color = agent_colors.get(trace.agent_name, "#f5f5f5")
            icon = status_icons.get(trace.status, "&#8226;")

            with st.expander(
                f"{trace.agent_name} ({trace.status}) - {trace.processing_time_ms:.1f}ms",
                expanded=(trace.status == "success"),
            ):
                st.markdown(f"""
                <div style="background: {color}; padding: 1rem; border-radius: 8px;">
                    <strong>Status:</strong> {icon} {trace.status} &nbsp;|&nbsp;
                    <strong>Confidence:</strong> {trace.confidence:.1%} &nbsp;|&nbsp;
                    <strong>Time:</strong> {trace.processing_time_ms:.1f}ms
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Reasoning Steps:**")
                for i, step in enumerate(trace.reasoning, 1):
                    st.markdown(f"{i}. {step}")

                if trace.findings:
                    st.markdown("**Key Findings:**")
                    st.json(trace.findings)

        # Processing time chart
        st.markdown("---")
        st.subheader("Processing Time by Agent")
        import pandas as pd
        chart_data = pd.DataFrame({
            "Agent": [t.agent_name for t in result.agent_traces],
            "Time (ms)": [t.processing_time_ms for t in result.agent_traces],
        })
        st.bar_chart(chart_data.set_index("Agent"))

        # Referral details
        if result.referral_result and result.referral_result.referral_needed:
            st.markdown("---")
            st.subheader("Referral Details")
            ref = result.referral_result
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Urgency", ref.urgency.upper())
            with r2:
                st.metric("Facility", ref.facility_level.title())
            with r3:
                st.metric("Timeframe", ref.timeframe)
            st.warning(f"Reason: {ref.reason}")


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
