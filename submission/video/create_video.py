"""
Create NEXUS demo video from captured screenshots.
Uses PIL for title slides and ffmpeg for video assembly.
Target: 3 minutes or less.
"""
import os
import subprocess
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

VIDEO_DIR = Path(__file__).parent
FRAMES_DIR = VIDEO_DIR / "frames"
OUTPUT_DIR = VIDEO_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Video settings
WIDTH, HEIGHT = 1440, 810
FPS = 1  # 1 frame per second for slideshow
BG_COLOR = (15, 23, 42)  # Dark blue
ACCENT_COLOR = (59, 130, 246)  # Blue
TEXT_COLOR = (255, 255, 255)
SUBTITLE_COLOR = (148, 163, 184)
SUCCESS_COLOR = (34, 197, 94)


def get_font(size):
    """Try to get a nice font, fall back to default."""
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def get_bold_font(size):
    """Try to get a bold font."""
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return get_font(size)


def create_title_slide(title, subtitle="", bullets=None, duration_seconds=5):
    """Create a title slide image."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title
    title_font = get_bold_font(52)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    y_start = 200 if not bullets else 120
    draw.text(((WIDTH - tw) // 2, y_start), title, fill=TEXT_COLOR, font=title_font)

    # Subtitle
    if subtitle:
        sub_font = get_font(28)
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox[2] - bbox[0]
        draw.text(((WIDTH - sw) // 2, y_start + 80), subtitle, fill=SUBTITLE_COLOR, font=sub_font)

    # Bullets
    if bullets:
        bullet_font = get_font(24)
        y = y_start + 140
        for bullet in bullets:
            draw.text((200, y), f"  {bullet}", fill=SUBTITLE_COLOR, font=bullet_font)
            y += 45

    # Bottom accent line
    draw.rectangle([(0, HEIGHT - 4), (WIDTH, HEIGHT)], fill=ACCENT_COLOR)

    return img, duration_seconds


def create_section_slide(title, model_badge="", duration_seconds=3):
    """Create a section transition slide."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (20, 30, 55))
    draw = ImageDraw.Draw(img)

    title_font = get_bold_font(44)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 320), title, fill=TEXT_COLOR, font=title_font)

    if model_badge:
        badge_font = get_font(22)
        bbox = draw.textbbox((0, 0), model_badge, font=badge_font)
        bw = bbox[2] - bbox[0]
        # Draw badge background
        bx = (WIDTH - bw - 20) // 2
        draw.rounded_rectangle([(bx, 400), (bx + bw + 20, 435)], radius=5, fill=ACCENT_COLOR)
        draw.text((bx + 10, 403), model_badge, fill=TEXT_COLOR, font=badge_font)

    draw.rectangle([(0, HEIGHT - 4), (WIDTH, HEIGHT)], fill=ACCENT_COLOR)
    return img, duration_seconds


def resize_screenshot(path):
    """Resize a screenshot to video dimensions."""
    img = Image.open(path)
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    return img


def build_video():
    """Build the complete video."""
    slides = []  # List of (image, duration_in_seconds)

    # === INTRO (0:00 - 0:15) ===
    intro, dur = create_title_slide(
        "NEXUS",
        "AI-Powered Maternal-Neonatal Assessment Platform",
        bullets=[
            "Non-invasive screening for anemia, jaundice, and birth asphyxia",
            "Built with Google HAI-DEF: MedSigLIP + HeAR + MedGemma",
            "6-Agent Clinical Workflow with WHO IMNCI Protocol Alignment",
            "Edge AI Ready: 7.31x compression for low-resource deployment",
        ],
        duration_seconds=8,
    )
    slides.append((intro, dur))

    # Problem statement
    problem, dur = create_title_slide(
        "The Problem",
        "Every day, 800 women and 7,400 newborns die from preventable causes",
        bullets=[
            "94% of deaths occur in low-resource settings",
            "6.9 million Community Health Workers lack diagnostic tools",
            "Maternal anemia affects 40% of pregnancies globally",
            "Neonatal jaundice affects 60% of newborns",
            "Birth asphyxia accounts for 23% of neonatal deaths",
        ],
        duration_seconds=7,
    )
    slides.append((problem, dur))

    # === HAI-DEF MODELS INFO (0:15 - 0:25) ===
    section, dur = create_section_slide("HAI-DEF Models Integration", "MedSigLIP + HeAR + MedGemma")
    slides.append((section, dur))

    slides.append((resize_screenshot(FRAMES_DIR / "02_haidef_models.png"), 6))

    # === ANEMIA SCREENING (0:25 - 0:40) ===
    section, dur = create_section_slide("Maternal Anemia Screening", "MedSigLIP")
    slides.append((section, dur))

    slides.append((resize_screenshot(FRAMES_DIR / "03_anemia_with_results.png"), 7))

    # Anemia results slide (text)
    anemia_results, dur = create_title_slide(
        "Anemia Detection Results",
        "MedSigLIP embeddings + trained SVM classifier",
        bullets=[
            "99.94% accuracy (5-fold cross-validation)",
            "218 images expanded to 1,744 with 7x augmentation",
            "Zero-shot + trained classifier ensemble",
            "Hemoglobin estimation from conjunctiva color features",
        ],
        duration_seconds=6,
    )
    slides.append((anemia_results, dur))

    # === JAUNDICE DETECTION (0:40 - 0:55) ===
    section, dur = create_section_slide("Neonatal Jaundice Detection", "MedSigLIP")
    slides.append((section, dur))

    slides.append((resize_screenshot(FRAMES_DIR / "04_jaundice_with_results.png"), 7))

    jaundice_results, dur = create_title_slide(
        "Jaundice Detection + Bilirubin Regression",
        "Novel: Continuous bilirubin prediction from frozen MedSigLIP embeddings",
        bullets=[
            "Classification: 96.73% accuracy (SVM on 2,235 images)",
            "Bilirubin Regression: MAE 2.564 mg/dL, Pearson r = 0.78",
            "3-layer MLP with BatchNorm on frozen embeddings",
            "Reduces need for blood draws in resource-limited settings",
        ],
        duration_seconds=7,
    )
    slides.append((jaundice_results, dur))

    # === CRY ANALYSIS (0:55 - 1:20) ===
    section, dur = create_section_slide("Infant Cry Analysis", "HeAR")
    slides.append((section, dur))

    slides.append((resize_screenshot(FRAMES_DIR / "06_cry_with_results.png"), 7))

    cry_results, dur = create_title_slide(
        "Cry Analysis Results",
        "HeAR 512-dim embeddings + trained 5-class cry classifier",
        bullets=[
            "83.81% accuracy (5-fold CV, 457 samples)",
            "5 cry types: hungry, belly pain, burping, discomfort, tired",
            "Asphyxia risk derived from distress patterns",
            "Real-time acoustic feature extraction (F0, spectral centroid)",
        ],
        duration_seconds=6,
    )
    slides.append((cry_results, dur))

    # === AGENTIC WORKFLOW (1:20 - 2:20) ===
    section, dur = create_section_slide("6-Agent Agentic Clinical Workflow", "MedSigLIP + HeAR + MedGemma")
    slides.append((section, dur))

    slides.append((resize_screenshot(FRAMES_DIR / "07_agentic_tab_ui.png"), 5))
    slides.append((resize_screenshot(FRAMES_DIR / "08_agentic_results_top.png"), 4))
    slides.append((resize_screenshot(FRAMES_DIR / "09_agentic_who_classification.png"), 6))
    slides.append((resize_screenshot(FRAMES_DIR / "10_agentic_reasoning_traces.png"), 6))
    slides.append((resize_screenshot(FRAMES_DIR / "11_agentic_chart_summary.png"), 6))

    # Agentic explanation
    agentic_explain, dur = create_title_slide(
        "Agentic Workflow Architecture",
        "6 sequential agents with step-by-step reasoning traces",
        bullets=[
            "1. Triage Agent: Danger sign scoring, demographic risk assessment",
            "2. Image Agent: MedSigLIP classification + bilirubin regression",
            "3. Audio Agent: HeAR cry analysis + asphyxia risk scoring",
            "4. Protocol Agent: WHO IMNCI protocols, comorbidity analysis",
            "5. Referral Agent: Facility matching, pre-referral stabilization",
            "6. Synthesis Agent: MedGemma unified clinical recommendations",
        ],
        duration_seconds=8,
    )
    slides.append((agentic_explain, dur))

    # === EDGE AI (2:20 - 2:40) ===
    section, dur = create_section_slide("Edge AI Deployment", "Offline-capable for low-resource settings")
    slides.append((section, dur))

    edge, dur = create_title_slide(
        "Edge AI: Smartphone Deployment",
        "INT8 dynamic quantization for 2 GB RAM devices",
        bullets=[
            "MedSigLIP: 812.6 MB -> 111.2 MB (7.31x compression)",
            "Pre-computed text embeddings: 12 KB (no text encoder on device)",
            "Total on-device: ~289 MB (fits Android 8.0+, ARM Cortex-A53)",
            "CPU inference: ~98ms per image on mobile processors",
        ],
        duration_seconds=7,
    )
    slides.append((edge, dur))

    # === CLOSING (2:40 - 3:00) ===
    closing, dur = create_title_slide(
        "NEXUS: Impact at Scale",
        "Empowering Community Health Workers worldwide",
        bullets=[
            "6.9M CHWs globally, 500M+ potential patient interactions",
            "3 HAI-DEF models: MedSigLIP + HeAR + MedGemma",
            "Novel bilirubin regression from frozen embeddings (r=0.78)",
            "Open source: github.com/Shahabul87/nexus",
            "Live demo: huggingface.co/spaces/Shahabul/nexus",
        ],
        duration_seconds=8,
    )
    slides.append((closing, dur))

    # Thank you
    thanks, dur = create_title_slide(
        "Thank You",
        "Built with Google HAI-DEF for the MedGemma Impact Challenge 2026",
        duration_seconds=5,
    )
    slides.append((thanks, dur))

    # === ASSEMBLE VIDEO ===
    print(f"Total slides: {len(slides)}")
    total_duration = sum(d for _, d in slides)
    print(f"Total duration: {total_duration} seconds ({total_duration/60:.1f} minutes)")

    # Save individual frames with durations for ffmpeg concat
    frame_list_path = OUTPUT_DIR / "frames.txt"
    frame_idx = 0
    with open(frame_list_path, "w") as f:
        for img, duration in slides:
            frame_path = OUTPUT_DIR / f"frame_{frame_idx:03d}.png"
            img.save(str(frame_path))
            f.write(f"file '{frame_path.name}'\n")
            f.write(f"duration {duration}\n")
            frame_idx += 1
        # Repeat last frame (ffmpeg concat demuxer quirk)
        f.write(f"file '{OUTPUT_DIR / f'frame_{frame_idx-1:03d}.png'}'\\n")

    print(f"Saved {frame_idx} frames to {OUTPUT_DIR}")

    # Build video with ffmpeg
    output_path = VIDEO_DIR / "NEXUS_Demo_Video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(frame_list_path),
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        str(output_path),
    ]
    print(f"\nRunning ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(OUTPUT_DIR))
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[-500:]}")
    else:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nVideo created: {output_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"Duration: {total_duration} seconds")

    return output_path


if __name__ == "__main__":
    build_video()
