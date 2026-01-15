#!/usr/bin/env python3
"""
Test Zero-Shot HAI-DEF Models

Tests MedSigLIP zero-shot classification on anemia and jaundice datasets.
Per NEXUS_MASTER_PLAN.md Week 1: Zero-Shot Validation.
"""

# Fix SSL certificate issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus.anemia_detector import AnemiaDetector
from nexus.jaundice_detector import JaundiceDetector
from nexus.cry_analyzer import CryAnalyzer
from nexus.clinical_synthesizer import ClinicalSynthesizer

import json
from datetime import datetime


def test_anemia_detector():
    """Test anemia detector on Eyes-Defy-Anemia dataset."""
    print("\n" + "=" * 60)
    print("TESTING ANEMIA DETECTOR (MedSigLIP)")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "raw" / "eyes-defy-anemia"

    if not data_dir.exists():
        print(f"Dataset not found at {data_dir}")
        return None

    # Find all images
    images = []
    for region in ["India", "Italy"]:
        region_dir = data_dir / region
        if region_dir.exists():
            images.extend(list(region_dir.rglob("*.jpg")))

    print(f"Found {len(images)} images")

    if len(images) == 0:
        print("No images found!")
        return None

    # Initialize detector
    print("\nInitializing Anemia Detector...")
    try:
        detector = AnemiaDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None

    # Test on sample images
    print(f"\nTesting on {min(10, len(images))} sample images...")
    results = []

    for img_path in images[:10]:
        try:
            result = detector.detect(img_path)
            color_info = detector.analyze_color_features(img_path)

            print(f"\n  Image: {img_path.name}")
            print(f"  Anemia Detected: {result['is_anemic']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Est. Hemoglobin: {color_info['estimated_hemoglobin']} g/dL")

            results.append({
                "image": str(img_path.name),
                "is_anemic": result["is_anemic"],
                "confidence": result["confidence"],
                "risk_level": result["risk_level"],
                "anemia_score": result["anemia_score"],
            })
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    return results


def test_jaundice_detector():
    """Test jaundice detector on NeoJaundice dataset."""
    print("\n" + "=" * 60)
    print("TESTING JAUNDICE DETECTOR (MedSigLIP)")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "raw" / "neojaundice" / "images"

    if not data_dir.exists():
        print(f"Dataset not found at {data_dir}")
        # Try alternate path
        alt_dir = Path(__file__).parent.parent / "data" / "raw" / "neojaundice"
        if alt_dir.exists():
            data_dir = alt_dir
        else:
            return None

    # Find all images
    images = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
    print(f"Found {len(images)} images")

    if len(images) == 0:
        print("No images found!")
        return None

    # Initialize detector
    print("\nInitializing Jaundice Detector...")
    try:
        detector = JaundiceDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None

    # Test on sample images
    print(f"\nTesting on {min(10, len(images))} sample images...")
    results = []

    for img_path in images[:10]:
        try:
            result = detector.detect(img_path)
            zone_info = detector.analyze_kramer_zones(img_path)

            print(f"\n  Image: {img_path.name}")
            print(f"  Jaundice Detected: {result['has_jaundice']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Severity: {result['severity']}")
            print(f"  Est. Bilirubin: {result['estimated_bilirubin']} mg/dL")
            print(f"  Kramer Zone: {zone_info['kramer_zone']}")

            results.append({
                "image": str(img_path.name),
                "has_jaundice": result["has_jaundice"],
                "confidence": result["confidence"],
                "severity": result["severity"],
                "estimated_bilirubin": result["estimated_bilirubin"],
            })
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    return results


def test_cry_analyzer():
    """Test cry analyzer on available audio datasets."""
    print("\n" + "=" * 60)
    print("TESTING CRY ANALYZER (HeAR)")
    print("=" * 60)

    # Check available datasets
    data_dirs = [
        Path(__file__).parent.parent / "data" / "raw" / "donate-a-cry" / "donateacry_corpus_cleaned_and_updated_data",
        Path(__file__).parent.parent / "data" / "raw" / "infant-cry-dataset" / "cry",
        Path(__file__).parent.parent / "data" / "raw" / "cryceleb" / "audio",
    ]

    audio_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            audio_files.extend(list(data_dir.rglob("*.wav"))[:5])

    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("No audio files found!")
        return None

    # Initialize analyzer
    print("\nInitializing Cry Analyzer...")
    try:
        analyzer = CryAnalyzer()
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        return None

    # Test on sample audio files
    print(f"\nTesting on {min(5, len(audio_files))} sample files...")
    results = []

    for audio_path in audio_files[:5]:
        try:
            result = analyzer.analyze(audio_path)

            print(f"\n  Audio: {audio_path.name}")
            print(f"  Abnormal Cry: {result['is_abnormal']}")
            print(f"  Asphyxia Risk: {result['asphyxia_risk']:.1%}")
            print(f"  Cry Type: {result['cry_type']}")
            print(f"  F0 Mean: {result['features']['f0_mean']:.1f} Hz")

            results.append({
                "audio": str(audio_path.name),
                "is_abnormal": result["is_abnormal"],
                "asphyxia_risk": result["asphyxia_risk"],
                "cry_type": result["cry_type"],
            })
        except Exception as e:
            print(f"  Error processing {audio_path.name}: {e}")

    return results


def test_clinical_synthesizer():
    """Test clinical synthesizer with sample findings."""
    print("\n" + "=" * 60)
    print("TESTING CLINICAL SYNTHESIZER (MedGemma)")
    print("=" * 60)

    # Initialize synthesizer (rule-based mode for testing)
    print("\nInitializing Clinical Synthesizer...")
    try:
        synthesizer = ClinicalSynthesizer(use_medgemma=False)
    except Exception as e:
        print(f"Error initializing synthesizer: {e}")
        return None

    # Test scenarios
    scenarios = [
        {
            "name": "Healthy Patient",
            "findings": {
                "anemia": {"is_anemic": False, "confidence": 0.85, "risk_level": "low"},
                "jaundice": {"has_jaundice": False, "confidence": 0.90, "severity": "none"},
                "cry": {"is_abnormal": False, "asphyxia_risk": 0.1, "cry_type": "hunger"},
            }
        },
        {
            "name": "Moderate Anemia",
            "findings": {
                "anemia": {"is_anemic": True, "confidence": 0.75, "risk_level": "medium", "estimated_hemoglobin": 9.0},
                "jaundice": {"has_jaundice": False, "confidence": 0.85, "severity": "none"},
                "cry": {"is_abnormal": False, "asphyxia_risk": 0.15, "cry_type": "discomfort"},
            }
        },
        {
            "name": "Severe Jaundice",
            "findings": {
                "anemia": {"is_anemic": False, "confidence": 0.80, "risk_level": "low"},
                "jaundice": {"has_jaundice": True, "confidence": 0.90, "severity": "severe", "needs_phototherapy": True, "estimated_bilirubin": 18.5},
                "cry": {"is_abnormal": False, "asphyxia_risk": 0.2, "cry_type": "tired"},
            }
        },
        {
            "name": "Multiple Concerns",
            "findings": {
                "anemia": {"is_anemic": True, "confidence": 0.70, "risk_level": "high", "estimated_hemoglobin": 6.5},
                "jaundice": {"has_jaundice": True, "confidence": 0.85, "severity": "moderate", "needs_phototherapy": False, "estimated_bilirubin": 12.0},
                "cry": {"is_abnormal": True, "asphyxia_risk": 0.65, "cry_type": "pain"},
            }
        },
    ]

    results = []

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}")
        try:
            result = synthesizer.synthesize(scenario["findings"])

            print(f"  Severity Level: {result.get('severity_level', 'N/A')}")
            print(f"  Referral Needed: {result.get('referral_needed', 'N/A')}")
            print(f"  Summary: {result['summary'][:100]}...")

            results.append({
                "scenario": scenario["name"],
                "severity_level": result.get("severity_level"),
                "referral_needed": result.get("referral_needed"),
            })
        except Exception as e:
            print(f"  Error: {e}")

    return results


def main():
    """Run all tests and save results."""
    print("\n" + "=" * 60)
    print("NEXUS HAI-DEF MODEL VALIDATION")
    print("Week 1: Zero-Shot Testing")
    print("=" * 60)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Test each component
    anemia_results = test_anemia_detector()
    if anemia_results:
        all_results["tests"]["anemia"] = anemia_results

    jaundice_results = test_jaundice_detector()
    if jaundice_results:
        all_results["tests"]["jaundice"] = jaundice_results

    cry_results = test_cry_analyzer()
    if cry_results:
        all_results["tests"]["cry"] = cry_results

    synthesizer_results = test_clinical_synthesizer()
    if synthesizer_results:
        all_results["tests"]["synthesizer"] = synthesizer_results

    # Save results
    results_dir = Path(__file__).parent.parent / "models" / "validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "zero_shot_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
