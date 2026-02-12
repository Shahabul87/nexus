"""Create card and thumbnail images for Kaggle writeup submission."""
from PIL import Image, ImageDraw, ImageFont
import os

def get_font(size):
    paths = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]
    for fp in paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

def get_bold_font(size):
    paths = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf"]
    for fp in paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return get_font(size)


def draw_centered(draw, text, y, font, fill, width):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, y), text, fill=fill, font=font)


def create_card(path, w=800, h=418):
    """Card image for Kaggle writeup."""
    img = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Gradient-like accent bar at top
    for i in range(6):
        color = (59 - i*5, 130 - i*10, 246 - i*10)
        draw.rectangle([(0, i), (w, i+1)], fill=color)

    # Title
    title_font = get_bold_font(56)
    draw_centered(draw, "NEXUS", 60, title_font, (255, 255, 255), w)

    # Subtitle
    sub_font = get_font(22)
    draw_centered(draw, "AI-Powered Maternal-Neonatal Screening Platform", 130, sub_font, (148, 163, 184), w)

    # Divider line
    draw.rectangle([(w//2 - 100, 175), (w//2 + 100, 177)], fill=(59, 130, 246))

    # Three model badges
    badge_font = get_bold_font(16)
    label_font = get_font(13)
    models = [
        ("MedSigLIP", "Image Analysis", (34, 197, 94)),
        ("HeAR", "Cry Analysis", (251, 191, 36)),
        ("MedGemma", "Clinical Reasoning", (139, 92, 246)),
    ]
    badge_y = 200
    badge_w = 200
    gap = 30
    start_x = (w - (badge_w * 3 + gap * 2)) // 2

    for i, (name, task, color) in enumerate(models):
        bx = start_x + i * (badge_w + gap)
        # Badge background
        draw.rounded_rectangle([(bx, badge_y), (bx + badge_w, badge_y + 55)], radius=8, fill=(30, 41, 59))
        draw.rounded_rectangle([(bx, badge_y), (bx + badge_w, badge_y + 55)], radius=8, outline=color, width=2)
        # Model name
        bbox = draw.textbbox((0, 0), name, font=badge_font)
        tw = bbox[2] - bbox[0]
        draw.text((bx + (badge_w - tw) // 2, badge_y + 8), name, fill=color, font=badge_font)
        # Task label
        bbox = draw.textbbox((0, 0), task, font=label_font)
        tw = bbox[2] - bbox[0]
        draw.text((bx + (badge_w - tw) // 2, badge_y + 32), task, fill=(148, 163, 184), font=label_font)

    # Key stats
    stat_font = get_bold_font(28)
    stat_label_font = get_font(13)
    stats = [
        ("6", "Agent Pipeline"),
        ("r=0.78", "Bilirubin Regression"),
        ("7.31x", "Edge Compression"),
    ]
    stat_y = 290
    stat_w = 180
    stat_start = (w - (stat_w * 3 + gap * 2)) // 2

    for i, (val, label) in enumerate(stats):
        sx = stat_start + i * (stat_w + gap)
        # Value
        bbox = draw.textbbox((0, 0), val, font=stat_font)
        tw = bbox[2] - bbox[0]
        draw.text((sx + (stat_w - tw) // 2, stat_y), val, fill=(255, 255, 255), font=stat_font)
        # Label
        bbox = draw.textbbox((0, 0), label, font=stat_label_font)
        tw = bbox[2] - bbox[0]
        draw.text((sx + (stat_w - tw) // 2, stat_y + 38), label, fill=(148, 163, 184), font=stat_label_font)

    # Bottom tagline
    tag_font = get_font(14)
    draw_centered(draw, "MedGemma Impact Challenge 2026", h - 40, tag_font, (100, 116, 139), w)

    # Bottom accent bar
    draw.rectangle([(0, h - 4), (w, h)], fill=(59, 130, 246))

    img.save(path)
    print(f"Card saved: {path} ({w}x{h})")


def create_thumbnail(path, w=560, h=280):
    """Thumbnail image for Kaggle writeup (560x280)."""
    img = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Top accent
    draw.rectangle([(0, 0), (w, 4)], fill=(59, 130, 246))

    # Title
    title_font = get_bold_font(44)
    draw_centered(draw, "NEXUS", 30, title_font, (255, 255, 255), w)

    # Subtitle
    sub_font = get_font(16)
    draw_centered(draw, "AI-Powered Maternal-Neonatal Screening", 85, sub_font, (148, 163, 184), w)

    # Divider
    draw.rectangle([(w//2 - 60, 115), (w//2 + 60, 117)], fill=(59, 130, 246))

    # Three model badges in a row
    badge_font = get_bold_font(14)
    label_font = get_font(11)
    models = [
        ("MedSigLIP", "Image", (34, 197, 94)),
        ("HeAR", "Audio", (251, 191, 36)),
        ("MedGemma", "Reasoning", (139, 92, 246)),
    ]
    badge_w = 145
    gap = 20
    start_x = (w - (badge_w * 3 + gap * 2)) // 2
    badge_y = 132

    for i, (name, task, color) in enumerate(models):
        bx = start_x + i * (badge_w + gap)
        draw.rounded_rectangle([(bx, badge_y), (bx + badge_w, badge_y + 42)], radius=6, fill=(30, 41, 59))
        draw.rounded_rectangle([(bx, badge_y), (bx + badge_w, badge_y + 42)], radius=6, outline=color, width=2)
        bbox = draw.textbbox((0, 0), name, font=badge_font)
        tw = bbox[2] - bbox[0]
        draw.text((bx + (badge_w - tw) // 2, badge_y + 5), name, fill=color, font=badge_font)
        bbox = draw.textbbox((0, 0), task, font=label_font)
        tw = bbox[2] - bbox[0]
        draw.text((bx + (badge_w - tw) // 2, badge_y + 24), task, fill=(148, 163, 184), font=label_font)

    # Stats row
    stat_font = get_bold_font(22)
    stat_label_font = get_font(11)
    stats = [("6-Agent", "Pipeline"), ("r=0.78", "Bilirubin"), ("7.31x", "Compression")]
    stat_w = 145
    stat_start = (w - (stat_w * 3 + gap * 2)) // 2
    stat_y = 195

    for i, (val, label) in enumerate(stats):
        sx = stat_start + i * (stat_w + gap)
        bbox = draw.textbbox((0, 0), val, font=stat_font)
        tw = bbox[2] - bbox[0]
        draw.text((sx + (stat_w - tw) // 2, stat_y), val, fill=(255, 255, 255), font=stat_font)
        bbox = draw.textbbox((0, 0), label, font=stat_label_font)
        tw = bbox[2] - bbox[0]
        draw.text((sx + (stat_w - tw) // 2, stat_y + 28), label, fill=(148, 163, 184), font=stat_label_font)

    # Bottom accent
    draw.rectangle([(0, h - 4), (w, h)], fill=(59, 130, 246))

    img.save(path)
    print(f"Thumbnail saved: {path} ({w}x{h})")


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    create_card(os.path.join(out, "kaggle_card.png"))
    create_thumbnail(os.path.join(out, "kaggle_thumbnail.png"))
