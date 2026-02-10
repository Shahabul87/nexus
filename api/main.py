"""
NEXUS API Server

FastAPI backend for the NEXUS Mobile App.
Serves HAI-DEF models for maternal-neonatal health screening.

HAI-DEF Models:
- MedSigLIP: Image analysis (anemia, jaundice)
- HeAR: Audio analysis (cry/asphyxia)
- MedGemma: Clinical synthesis

API Version: 1.0.0
"""

import sys
from pathlib import Path

# Add src to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import base64
import tempfile
import logging
import time
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("nexus-api")

# Import NEXUS models
try:
    from nexus.anemia_detector import AnemiaDetector
    from nexus.jaundice_detector import JaundiceDetector
    from nexus.cry_analyzer import CryAnalyzer
    from nexus.clinical_synthesizer import ClinicalSynthesizer
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="NEXUS API",
    description="AI-Powered Maternal-Neonatal Care Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for mobile app
import os as _os
# Default to localhost origins; set CORS_ORIGINS env var for production
_cors_default = "http://localhost:3000,http://localhost:8081"
_cors_origins = _os.environ.get("CORS_ORIGINS", _cors_default).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
anemia_detector: Optional[AnemiaDetector] = None
jaundice_detector: Optional[JaundiceDetector] = None
cry_analyzer: Optional[CryAnalyzer] = None
clinical_synthesizer: Optional[ClinicalSynthesizer] = None

# Paths to trained classifier checkpoints
CHECKPOINT_DIR = Path(__file__).parent.parent / "models" / "checkpoints"
LINEAR_PROBE_DIR = Path(__file__).parent.parent / "models" / "linear_probes"


def _get_model_for_classifier_dimensions(n_features: int) -> str:
    """
    Get the model name that matches the classifier's expected feature dimensions.

    Different SigLIP/MedSigLIP models produce different embedding sizes:
    - medsiglip-448: varies (check model card)
    - siglip-base-patch16-224: 768 features
    """
    dimension_to_model = {
        768: "google/siglip-base-patch16-224",
    }
    # MedSigLIP-448 is the default for new probes - add its dimension when known
    return dimension_to_model.get(n_features, "google/medsiglip-448")


def _load_classifier_for_detector(detector, model_type: str) -> None:
    """
    Load trained classifier into detector if available.

    Checks for linear probes (.joblib) first, then PyTorch checkpoints (.pt).
    """
    # Try linear probe first (sklearn)
    probe_path = LINEAR_PROBE_DIR / f"{model_type}_linear_probe.joblib"
    if probe_path.exists():
        try:
            import joblib
            classifier = joblib.load(probe_path)
            detector.classifier = classifier
            print(f"Loaded linear probe classifier for {model_type} from {probe_path}")
            return
        except Exception as e:
            print(f"Warning: Failed to load linear probe for {model_type}: {e}")

    # Try PyTorch checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"{model_type}_best.pt"
    if checkpoint_path.exists():
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'classifier' in checkpoint:
                detector.classifier = checkpoint['classifier']
                print(f"Loaded PyTorch classifier for {model_type} from {checkpoint_path}")
                return
        except Exception as e:
            print(f"Warning: Failed to load checkpoint for {model_type}: {e}")

    print(f"No trained classifier found for {model_type}, using zero-shot")


def _get_model_name_for_trained_classifier(model_type: str) -> Optional[str]:
    """
    Determine the correct model to use based on a trained classifier's dimensions.

    Returns the model name that matches the classifier's expected input dimensions,
    or None if no trained classifier exists.
    """
    probe_path = LINEAR_PROBE_DIR / f"{model_type}_linear_probe.joblib"
    if probe_path.exists():
        try:
            import joblib
            classifier = joblib.load(probe_path)
            n_features = classifier.n_features_in_
            model_name = _get_model_for_classifier_dimensions(n_features)
            if model_name:
                print(f"Trained classifier expects {n_features} features, using {model_name}")
                return model_name
        except Exception as e:
            print(f"Warning: Could not determine classifier dimensions: {e}")
    return None


def get_anemia_detector() -> AnemiaDetector:
    """Get or create anemia detector instance with trained classifier."""
    global anemia_detector
    if anemia_detector is None:
        # Check if we have a trained classifier and get the matching model
        model_name = _get_model_name_for_trained_classifier("anemia")
        anemia_detector = AnemiaDetector(model_name=model_name)
        _load_classifier_for_detector(anemia_detector, "anemia")
    return anemia_detector


def get_jaundice_detector() -> JaundiceDetector:
    """Get or create jaundice detector instance with trained classifier."""
    global jaundice_detector
    if jaundice_detector is None:
        # Check if we have a trained classifier and get the matching model
        model_name = _get_model_name_for_trained_classifier("jaundice")
        jaundice_detector = JaundiceDetector(model_name=model_name)
        _load_classifier_for_detector(jaundice_detector, "jaundice")
    return jaundice_detector


def get_cry_analyzer() -> CryAnalyzer:
    """Get or create cry analyzer instance with trained classifier."""
    global cry_analyzer
    if cry_analyzer is None:
        # Cry analyzer takes classifier_path in constructor
        probe_path = LINEAR_PROBE_DIR / "cry_linear_probe.joblib"
        classifier_path = str(probe_path) if probe_path.exists() else None
        cry_analyzer = CryAnalyzer(classifier_path=classifier_path)
    return cry_analyzer


def get_clinical_synthesizer() -> ClinicalSynthesizer:
    """Get or create clinical synthesizer instance.

    By default, tries to use MedGemma. Falls back to rule-based if unavailable.
    Set NEXUS_USE_MEDGEMMA=false env var to force rule-based synthesis.
    """
    global clinical_synthesizer
    if clinical_synthesizer is None:
        import os
        use_medgemma = os.environ.get("NEXUS_USE_MEDGEMMA", "true").lower() != "false"
        clinical_synthesizer = ClinicalSynthesizer(use_medgemma=use_medgemma)
    return clinical_synthesizer


# Request/Response Models with Validation
class AnemiaRequest(BaseModel):
    """Request model for anemia detection endpoint."""
    image: str = Field(..., description="Base64 encoded conjunctiva image")
    model: str = Field(default="medsiglip", description="HAI-DEF model to use")

    @validator("image")
    def validate_image(cls, v: str) -> str:
        if not v or len(v) < 100:
            raise ValueError("Invalid image data: too short to be a valid base64 image")
        return v


class AnemiaResponse(BaseModel):
    """Response model for anemia detection."""
    is_anemic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="low, medium, or high")
    estimated_hemoglobin: float = Field(..., ge=0.0, le=20.0)
    recommendation: str
    anemia_score: float = Field(..., ge=0.0, le=1.0)
    healthy_score: float = Field(..., ge=0.0, le=1.0)


class JaundiceRequest(BaseModel):
    """Request model for jaundice detection endpoint."""
    image: str = Field(..., description="Base64 encoded skin/sclera image")
    model: str = Field(default="medsiglip", description="HAI-DEF model to use")

    @validator("image")
    def validate_image(cls, v: str) -> str:
        if not v or len(v) < 100:
            raise ValueError("Invalid image data: too short to be a valid base64 image")
        return v


class JaundiceResponse(BaseModel):
    """Response model for jaundice detection."""
    has_jaundice: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="none, mild, moderate, severe, or critical")
    estimated_bilirubin: float = Field(..., ge=0.0, le=30.0)
    needs_phototherapy: bool
    recommendation: str
    kramer_zone: int = Field(..., ge=0, le=5)


class CryRequest(BaseModel):
    """Request model for cry analysis endpoint."""
    audio: str = Field(..., description="Base64 encoded WAV audio")
    model: str = Field(default="hear", description="HAI-DEF model to use")

    @validator("audio")
    def validate_audio(cls, v: str) -> str:
        if not v or len(v) < 100:
            raise ValueError("Invalid audio data: too short to be a valid base64 audio")
        return v


class CryResponse(BaseModel):
    """Response model for cry analysis."""
    is_abnormal: bool
    asphyxia_risk: float = Field(..., ge=0.0, le=1.0)
    cry_type: str
    risk_level: str = Field(..., description="low, medium, or high")
    recommendation: str
    features: dict


class CombinedRequest(BaseModel):
    """Request model for combined assessment endpoint."""
    conjunctiva_image: Optional[str] = Field(None, description="Base64 encoded conjunctiva image")
    skin_image: Optional[str] = Field(None, description="Base64 encoded skin image")
    cry_audio: Optional[str] = Field(None, description="Base64 encoded cry audio")
    synthesizer: str = Field(default="medgemma", description="Synthesizer model")

    @validator("conjunctiva_image", "skin_image", "cry_audio", pre=True)
    def validate_optional_data(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) < 100:
            raise ValueError("Invalid data: too short to be valid")
        return v


class CombinedResponse(BaseModel):
    """Response model for combined assessment."""
    summary: str
    severity_level: str = Field(..., description="GREEN, YELLOW, or RED")
    severity_description: str
    immediate_actions: List[str]
    referral_needed: bool
    referral_urgency: str
    follow_up: str
    urgent_conditions: List[str]
    model: str


class SynthesizeRequest(BaseModel):
    """Request model for clinical synthesis endpoint."""
    patient_type: str = Field(default="newborn", description="pregnant or newborn")
    danger_signs: List[str] = Field(default_factory=list)
    anemia_result: Optional[dict] = None
    jaundice_result: Optional[dict] = None
    cry_result: Optional[dict] = None
    model: str = Field(default="medgemma", description="Synthesizer model")

    @validator("patient_type")
    def validate_patient_type(cls, v: str) -> str:
        if v not in ["pregnant", "newborn"]:
            raise ValueError("patient_type must be 'pregnant' or 'newborn'")
        return v


class SynthesizeResponse(BaseModel):
    """Response model for clinical synthesis endpoint."""
    synthesis: str
    recommendation: str
    immediate_actions: List[str]
    severity: str = Field(..., description="GREEN, YELLOW, or RED")
    referral_needed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str


class ProtocolResponse(BaseModel):
    """Response model for WHO IMNCI protocol endpoint."""
    name: str
    source: str
    condition: str
    steps: List[str]
    referral_criteria: str
    warning_signs: List[str]


class AgentTraceResponse(BaseModel):
    """Response model for a single agent's reasoning trace."""
    agent_name: str
    status: str = Field(..., description="success, skipped, or error")
    reasoning: List[str] = Field(default_factory=list)
    findings: dict = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    processing_time_ms: float = Field(0.0, ge=0.0)


class AgenticRequest(BaseModel):
    """Request model for agentic workflow assessment endpoint."""
    patient_type: str = Field(default="newborn", description="pregnant or newborn")
    conjunctiva_image: Optional[str] = Field(None, description="Base64 encoded conjunctiva image")
    skin_image: Optional[str] = Field(None, description="Base64 encoded skin image")
    cry_audio: Optional[str] = Field(None, description="Base64 encoded cry audio")
    danger_signs: List[dict] = Field(default_factory=list, description="List of danger sign dicts")
    patient_info: Optional[dict] = Field(None, description="Patient information")

    @validator("patient_type")
    def validate_patient_type(cls, v: str) -> str:
        if v not in ["pregnant", "newborn"]:
            raise ValueError("patient_type must be 'pregnant' or 'newborn'")
        return v

    @validator("conjunctiva_image", "skin_image", "cry_audio", pre=True)
    def validate_optional_data(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) < 100:
            raise ValueError("Invalid data: too short to be valid")
        return v


class AgenticResponse(BaseModel):
    """Response model for agentic workflow assessment."""
    success: bool
    patient_type: str
    who_classification: str = Field(..., description="RED, YELLOW, or GREEN")
    clinical_synthesis: str
    recommendation: str
    immediate_actions: List[str]
    processing_time_ms: float
    timestamp: str
    triage: Optional[dict] = None
    referral: Optional[dict] = None
    protocol: Optional[dict] = None
    agent_traces: List[AgentTraceResponse] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: str
    request_id: Optional[str] = None


def cleanup_temp_file(path: str) -> None:
    """Safely clean up a temporary file."""
    import os
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass  # Ignore cleanup errors


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}"

    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        logger.info(f"[{request_id}] Completed in {process_time:.3f}s - Status: {response.status_code}")

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id

        return response
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Check server logs for details.",
            "timestamp": datetime.now().isoformat(),
        }
    )


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint — verifies model import availability."""
    models_loaded = {
        "anemia_detector": anemia_detector is not None,
        "jaundice_detector": jaundice_detector is not None,
        "cry_analyzer": cry_analyzer is not None,
        "clinical_synthesizer": clinical_synthesizer is not None,
    }
    any_loaded = any(models_loaded.values())
    return {
        "status": "healthy" if MODELS_AVAILABLE else "degraded",
        "models_available": MODELS_AVAILABLE,
        "models_loaded": models_loaded,
        "any_model_loaded": any_loaded,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


# Anemia Detection Endpoint
@app.post("/api/anemia/detect", response_model=AnemiaResponse)
async def detect_anemia(request: AnemiaRequest):
    """
    Detect anemia from conjunctiva image.

    Uses MedSigLIP for zero-shot classification.
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    tmp_path = None
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        # Run detection
        detector = get_anemia_detector()
        result = detector.detect(tmp_path)
        color_info = detector.analyze_color_features(tmp_path)

        return AnemiaResponse(
            is_anemic=result["is_anemic"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            estimated_hemoglobin=color_info["estimated_hemoglobin"],
            recommendation=result["recommendation"],
            anemia_score=result["anemia_score"],
            healthy_score=result["healthy_score"],
        )
    except RuntimeError as e:
        logger.error(f"Model runtime error in anemia detection: {e}")
        raise HTTPException(status_code=503, detail="Model inference unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


# Jaundice Detection Endpoint
@app.post("/api/jaundice/detect", response_model=JaundiceResponse)
async def detect_jaundice(request: JaundiceRequest):
    """
    Detect neonatal jaundice from skin/sclera image.

    Uses MedSigLIP for zero-shot classification with Kramer zone analysis.
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    tmp_path = None
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        # Run detection
        detector = get_jaundice_detector()
        result = detector.detect(tmp_path)
        zone_info = detector.analyze_kramer_zones(tmp_path)

        return JaundiceResponse(
            has_jaundice=result["has_jaundice"],
            confidence=result["confidence"],
            severity=result["severity"],
            estimated_bilirubin=result["estimated_bilirubin"],
            needs_phototherapy=result["needs_phototherapy"],
            recommendation=result["recommendation"],
            kramer_zone=zone_info["kramer_zone"],
        )
    except RuntimeError as e:
        logger.error(f"Model runtime error in jaundice detection: {e}")
        raise HTTPException(status_code=503, detail="Model inference unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


# Cry Analysis Endpoint
@app.post("/api/cry/analyze", response_model=CryResponse)
async def analyze_cry(request: CryRequest):
    """
    Analyze infant cry audio for asphyxia detection.

    Uses HeAR for health acoustic analysis.
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    tmp_path = None
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # Run analysis
        analyzer = get_cry_analyzer()
        result = analyzer.analyze(tmp_path)

        return CryResponse(
            is_abnormal=result["is_abnormal"],
            asphyxia_risk=result["asphyxia_risk"],
            cry_type=result["cry_type"],
            risk_level=result["risk_level"],
            recommendation=result["recommendation"],
            features=result["features"],
        )
    except RuntimeError as e:
        logger.error(f"Model runtime error in cry analysis: {e}")
        raise HTTPException(status_code=503, detail="Model inference unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


# Combined Assessment Endpoint
@app.post("/api/combined/assess", response_model=CombinedResponse)
async def combined_assessment(request: CombinedRequest):
    """
    Run comprehensive assessment using all available inputs.

    Uses MedGemma for clinical synthesis following WHO IMNCI protocols.
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    temp_files = []
    try:
        findings = {}

        # Analyze conjunctiva if provided
        if request.conjunctiva_image:
            try:
                image_data = base64.b64decode(request.conjunctiva_image)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name
                    temp_files.append(tmp_path)
                detector = get_anemia_detector()
                findings["anemia"] = detector.detect(tmp_path)
            except Exception as e:
                logger.warning(f"Anemia detection failed: {e}")

        # Analyze skin if provided
        if request.skin_image:
            try:
                image_data = base64.b64decode(request.skin_image)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name
                    temp_files.append(tmp_path)
                detector = get_jaundice_detector()
                findings["jaundice"] = detector.detect(tmp_path)
            except Exception as e:
                logger.warning(f"Jaundice detection failed: {e}")

        # Analyze cry if provided
        if request.cry_audio:
            try:
                audio_data = base64.b64decode(request.cry_audio)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name
                    temp_files.append(tmp_path)
                analyzer = get_cry_analyzer()
                findings["cry"] = analyzer.analyze(tmp_path)
            except Exception as e:
                logger.warning(f"Cry analysis failed: {e}")

        # Clinical synthesis
        synthesizer = get_clinical_synthesizer()
        synthesis = synthesizer.synthesize(findings)

        return CombinedResponse(
            summary=synthesis.get("summary", ""),
            severity_level=synthesis.get("severity_level", "GREEN"),
            severity_description=synthesis.get("severity_description", ""),
            immediate_actions=synthesis.get("immediate_actions", []),
            referral_needed=synthesis.get("referral_needed", False),
            referral_urgency=synthesis.get("referral_urgency", "none"),
            follow_up=synthesis.get("follow_up", ""),
            urgent_conditions=synthesis.get("urgent_conditions", []),
            model=synthesis.get("model", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for tmp_path in temp_files:
            cleanup_temp_file(tmp_path)


# Clinical Synthesis Endpoint
@app.post("/api/synthesize", response_model=SynthesizeResponse)
async def synthesize_findings(request: SynthesizeRequest):
    """
    Synthesize clinical findings using MedGemma.

    Takes individual findings (anemia, jaundice, cry analysis) and generates
    a clinical synthesis with recommendations following WHO IMNCI protocols.
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    try:
        # Build findings dict from request
        findings = {}

        if request.anemia_result:
            findings["anemia"] = request.anemia_result

        if request.jaundice_result:
            findings["jaundice"] = request.jaundice_result

        if request.cry_result:
            findings["cry"] = request.cry_result

        if request.danger_signs:
            findings["symptoms"] = ", ".join(request.danger_signs)

        findings["patient_info"] = {"type": request.patient_type}

        # Run clinical synthesis
        synthesizer = get_clinical_synthesizer()
        synthesis = synthesizer.synthesize(findings)

        # Map synthesis result to response
        return SynthesizeResponse(
            synthesis=synthesis.get("summary", ""),
            recommendation=synthesis.get("immediate_actions", ["Continue routine care"])[0],
            immediate_actions=synthesis.get("immediate_actions", []),
            severity=synthesis.get("severity_level", "GREEN"),
            referral_needed=synthesis.get("referral_needed", False),
            confidence=0.85 if synthesis.get("model", "").startswith("MedGemma") else 0.75,
            model=synthesis.get("model", "Unknown"),
        )
    except RuntimeError as e:
        logger.error(f"Model runtime error in synthesis: {e}")
        raise HTTPException(status_code=503, detail="Model inference unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agentic Workflow Assessment Endpoint
@app.post("/api/agentic/assess", response_model=AgenticResponse)
async def agentic_assessment(request: AgenticRequest):
    """
    Run comprehensive agentic workflow assessment with 6 specialized agents.

    Pipeline: Triage -> Image (MedSigLIP) -> Audio (HeAR) -> Protocol (WHO IMNCI)
              -> Referral -> Synthesis (MedGemma)

    Returns full workflow result with step-by-step reasoning traces from each agent.
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    temp_files = []
    try:
        from nexus.agentic_workflow import (
            AgenticWorkflowEngine,
            AgentPatientInfo,
            DangerSign,
            WorkflowInput,
        )

        # Build patient info
        info = AgentPatientInfo(patient_type=request.patient_type)
        if request.patient_info:
            info.patient_id = request.patient_info.get("patient_id", "")
            info.gestational_weeks = request.patient_info.get("gestational_weeks")
            info.birth_weight = request.patient_info.get("birth_weight")
            info.apgar_score = request.patient_info.get("apgar_score")
            info.age_hours = request.patient_info.get("age_hours")

        # Build danger signs
        signs = []
        for s in request.danger_signs:
            signs.append(DangerSign(
                id=s.get("id", ""),
                label=s.get("label", ""),
                severity=s.get("severity", "medium"),
                present=s.get("present", True),
            ))

        # Decode base64 inputs to temp files
        conjunctiva_path = None
        skin_path = None
        cry_path = None

        if request.conjunctiva_image:
            image_data = base64.b64decode(request.conjunctiva_image)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_data)
                conjunctiva_path = tmp.name
                temp_files.append(tmp.name)

        if request.skin_image:
            image_data = base64.b64decode(request.skin_image)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_data)
                skin_path = tmp.name
                temp_files.append(tmp.name)

        if request.cry_audio:
            audio_data = base64.b64decode(request.cry_audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                cry_path = tmp.name
                temp_files.append(tmp.name)

        workflow_input = WorkflowInput(
            patient_type=request.patient_type,
            patient_info=info,
            danger_signs=signs,
            conjunctiva_image=conjunctiva_path,
            skin_image=skin_path,
            cry_audio=cry_path,
        )

        # Create engine with lazily-loaded model instances (not the raw globals)
        engine = AgenticWorkflowEngine(
            anemia_detector=get_anemia_detector() if MODELS_AVAILABLE else None,
            jaundice_detector=get_jaundice_detector() if MODELS_AVAILABLE else None,
            cry_analyzer=get_cry_analyzer() if MODELS_AVAILABLE else None,
            synthesizer=get_clinical_synthesizer() if MODELS_AVAILABLE else None,
        )

        result = engine.execute(workflow_input)

        # Build response
        agent_traces = [
            AgentTraceResponse(
                agent_name=t.agent_name,
                status=t.status,
                reasoning=t.reasoning,
                findings=t.findings,
                confidence=t.confidence,
                processing_time_ms=t.processing_time_ms,
            )
            for t in result.agent_traces
        ]

        triage_dict = None
        if result.triage_result:
            triage_dict = {
                "risk_level": result.triage_result.risk_level,
                "score": result.triage_result.score,
                "critical_signs": result.triage_result.critical_signs,
                "immediate_referral": result.triage_result.immediate_referral_needed,
            }

        referral_dict = None
        if result.referral_result:
            referral_dict = {
                "referral_needed": result.referral_result.referral_needed,
                "urgency": result.referral_result.urgency,
                "facility_level": result.referral_result.facility_level,
                "reason": result.referral_result.reason,
                "timeframe": result.referral_result.timeframe,
            }

        protocol_dict = None
        if result.protocol_result:
            protocol_dict = {
                "classification": result.protocol_result.classification,
                "applicable_protocols": result.protocol_result.applicable_protocols,
                "treatment_recommendations": result.protocol_result.treatment_recommendations,
                "follow_up_schedule": result.protocol_result.follow_up_schedule,
            }

        return AgenticResponse(
            success=result.success,
            patient_type=result.patient_type,
            who_classification=result.who_classification,
            clinical_synthesis=result.clinical_synthesis,
            recommendation=result.recommendation,
            immediate_actions=result.immediate_actions,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.timestamp,
            triage=triage_dict,
            referral=referral_dict,
            protocol=protocol_dict,
            agent_traces=agent_traces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for tmp_path in temp_files:
            cleanup_temp_file(tmp_path)


# Model Info Endpoint
@app.get("/api/models")
async def get_models_info():
    """Get information about available HAI-DEF models."""
    # Get actual loaded model names if available
    anemia_model = "not_loaded"
    jaundice_model = "not_loaded"
    cry_model = "not_loaded"
    synthesizer_model = "not_loaded"

    if MODELS_AVAILABLE:
        try:
            if anemia_detector is not None:
                anemia_model = getattr(anemia_detector, 'model_name', 'google/medsiglip-448')
            if jaundice_detector is not None:
                jaundice_model = getattr(jaundice_detector, 'model_name', 'google/medsiglip-448')
            if cry_analyzer is not None:
                cry_model = "HeAR (PyTorch)" if getattr(cry_analyzer, '_hear_available', False) else "Acoustic Features"
            if clinical_synthesizer is not None:
                medgemma_available = getattr(clinical_synthesizer, '_medgemma_available', False)
                synthesizer_model = "MedGemma 4B" if medgemma_available else "Rule-based (WHO IMNCI)"
        except Exception:
            pass

    return {
        "hai_def_models": {
            "medsiglip": {
                "name": "MedSigLIP",
                "version": "google/medsiglip-448",
                "fallback_versions": [
                    "google/siglip-base-patch16-224"
                ],
                "loaded_model": anemia_model,
                "use_cases": ["anemia_detection", "jaundice_detection"],
                "method": "zero-shot_classification + linear_probe",
            },
            "hear": {
                "name": "HeAR",
                "version": "google/hear-pytorch",
                "loaded_model": cry_model,
                "use_cases": ["cry_analysis", "asphyxia_detection"],
                "method": "health_acoustic_embeddings + linear_probe",
                "fallback": "Acoustic feature extraction (deterministic)",
            },
            "medgemma": {
                "name": "MedGemma",
                "version": "google/medgemma-4b-it",
                "loaded_model": synthesizer_model,
                "use_cases": ["clinical_synthesis"],
                "method": "who_imnci_protocols",
                "fallback": "Rule-based synthesis (WHO IMNCI)",
            },
        },
        "status": "operational" if MODELS_AVAILABLE else "models_not_loaded",
    }


@app.get("/api/models/verify")
async def verify_models():
    """Verify which HAI-DEF models are actually loaded and operational."""
    verification = {
        "medsiglip": {
            "loaded": anemia_detector is not None,
            "model_name": getattr(anemia_detector, 'model_name', None) if anemia_detector else None,
            "is_hai_def": "medsiglip" in getattr(anemia_detector, 'model_name', '') if anemia_detector else False,
        },
        "hear": {
            "loaded": cry_analyzer is not None,
            "hear_available": getattr(cry_analyzer, '_hear_available', False) if cry_analyzer else False,
        },
        "medgemma": {
            "loaded": clinical_synthesizer is not None,
            "medgemma_available": getattr(clinical_synthesizer, '_medgemma_available', False) if clinical_synthesizer else False,
        },
    }
    all_hai_def = all([
        verification["medsiglip"]["is_hai_def"],
        verification["hear"]["hear_available"],
        verification["medgemma"]["medgemma_available"],
    ])
    verification["all_hai_def_active"] = all_hai_def
    return verification


# WHO IMNCI Protocol Endpoint
WHO_IMNCI_PROTOCOLS = {
    "anemia": {
        "name": "Maternal Anemia Management",
        "source": "WHO IMNCI Guidelines 2014",
        "condition": "anemia",
        "steps": [
            "Assess pallor of conjunctiva, palms, and nail beds",
            "If severe pallor: Urgent referral for blood transfusion",
            "If some pallor: Iron supplementation (60mg elemental iron + 400mcg folic acid daily)",
            "Counsel on iron-rich foods (dark leafy vegetables, meat, beans)",
            "De-worm if not done in last 6 months",
            "Test for malaria in endemic areas",
            "Follow up in 4 weeks to reassess",
        ],
        "referral_criteria": "Hb < 7 g/dL or severe pallor with symptoms (shortness of breath, fast heartbeat)",
        "warning_signs": [
            "Severe pallor",
            "Difficulty breathing",
            "Fast or difficult breathing",
            "Extreme fatigue",
            "Swelling of feet/face",
        ],
    },
    "jaundice": {
        "name": "Neonatal Jaundice Management",
        "source": "WHO IMNCI Guidelines 2014",
        "condition": "jaundice",
        "steps": [
            "Check for yellow skin/eyes - press skin to assess in natural light",
            "If jaundice in first 24 hours of life: URGENT referral (pathological)",
            "If jaundice extends below umbilicus: Consider phototherapy",
            "Ensure frequent breastfeeding (8-12 times per day)",
            "Keep baby warm but not overheated",
            "Expose to indirect sunlight (not direct) for short periods",
            "If bilirubin > 15 mg/dL: Start phototherapy",
            "If bilirubin > 20-25 mg/dL: Consider exchange transfusion",
        ],
        "referral_criteria": "Jaundice < 24 hours old, bilirubin > 20 mg/dL, or signs of acute bilirubin encephalopathy",
        "warning_signs": [
            "Jaundice appearing in first 24 hours",
            "Jaundice extending to palms and soles",
            "Dark urine or pale stools",
            "Poor feeding or lethargy",
            "High-pitched cry",
            "Arching of back (opisthotonus)",
        ],
    },
    "asphyxia": {
        "name": "Birth Asphyxia Management",
        "source": "WHO Neonatal Resuscitation Guidelines 2012",
        "condition": "asphyxia",
        "steps": [
            "Assess APGAR score at 1 and 5 minutes",
            "Dry the baby and provide warmth",
            "Clear airway if needed (suction only if meconium or blood visible)",
            "Stimulate by rubbing back or flicking soles",
            "If not breathing after 30 seconds: Begin bag-mask ventilation",
            "Provide 40 breaths per minute with room air",
            "If heart rate < 60 after 30 seconds of ventilation: Start chest compressions",
            "Refer for evaluation if abnormal cry, seizures, or poor feeding persists",
        ],
        "referral_criteria": "APGAR < 7 at 5 minutes, persistent abnormal cry, seizures, or inability to feed",
        "warning_signs": [
            "Not breathing at birth",
            "Gasping or weak breathing",
            "Floppy/limp baby",
            "Pale or blue skin color",
            "Heart rate < 100 at birth",
            "Seizures",
            "Abnormal high-pitched cry",
        ],
    },
    "danger_signs_newborn": {
        "name": "Newborn Danger Signs",
        "source": "WHO IMNCI Guidelines 2014",
        "condition": "danger_signs_newborn",
        "steps": [
            "Check for convulsions or history of convulsions",
            "Count respiratory rate (fast breathing > 60/min)",
            "Check for severe chest indrawing",
            "Check temperature (fever > 37.5°C or hypothermia < 35.5°C)",
            "Check umbilicus for redness extending to skin",
            "Count skin pustules (many = severe infection)",
            "Check movement (less than normal = danger)",
            "If any danger sign present: URGENT referral",
        ],
        "referral_criteria": "Any danger sign present requires urgent referral",
        "warning_signs": [
            "Not feeding well",
            "Convulsions",
            "Fast breathing (> 60/min)",
            "Severe chest indrawing",
            "High fever (> 37.5°C) or hypothermia (< 35.5°C)",
            "Umbilical redness extending to skin",
            "Many skin pustules",
            "Movement only when stimulated or no movement",
        ],
    },
    "danger_signs_pregnant": {
        "name": "Pregnancy Danger Signs",
        "source": "WHO Antenatal Care Guidelines",
        "condition": "danger_signs_pregnant",
        "steps": [
            "Ask about vaginal bleeding",
            "Check for severe headache with blurred vision (pre-eclampsia)",
            "Check for convulsions (eclampsia)",
            "Ask about fever",
            "Check for severe abdominal pain",
            "Ask about rupture of membranes",
            "Check for reduced or absent fetal movements",
            "If any danger sign present: URGENT referral",
        ],
        "referral_criteria": "Any danger sign requires urgent referral to health facility",
        "warning_signs": [
            "Vaginal bleeding",
            "Severe headache with blurred vision",
            "Convulsions/fits",
            "High fever",
            "Severe abdominal pain",
            "Difficulty breathing",
            "Water breaking before labor",
            "Reduced or absent baby movements",
        ],
    },
}


@app.get("/api/protocol/{condition}", response_model=ProtocolResponse)
async def get_protocol(condition: str):
    """
    Get WHO IMNCI protocol for a specific condition.

    Available conditions:
    - anemia: Maternal anemia management
    - jaundice: Neonatal jaundice management
    - asphyxia: Birth asphyxia management
    - danger_signs_newborn: Newborn danger signs
    - danger_signs_pregnant: Pregnancy danger signs
    """
    condition_lower = condition.lower().replace("-", "_").replace(" ", "_")

    if condition_lower not in WHO_IMNCI_PROTOCOLS:
        available = ", ".join(WHO_IMNCI_PROTOCOLS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Protocol not found for '{condition}'. Available: {available}"
        )

    protocol = WHO_IMNCI_PROTOCOLS[condition_lower]
    return ProtocolResponse(**protocol)


@app.get("/api/protocols")
async def list_protocols():
    """List all available WHO IMNCI protocols."""
    return {
        "protocols": [
            {
                "condition": key,
                "name": value["name"],
                "source": value["source"],
            }
            for key, value in WHO_IMNCI_PROTOCOLS.items()
        ],
        "count": len(WHO_IMNCI_PROTOCOLS),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting NEXUS API Server...")
    logger.info("HAI-DEF Models: MedSigLIP, HeAR, MedGemma")
    logger.info("Documentation: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
