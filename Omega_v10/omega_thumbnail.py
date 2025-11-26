# generate_omega_thumbnail.py
# Run this once → creates omega_thumbnail.png (1200×630)

from PIL import Image, ImageDraw, ImageFont
import os

# Create blank image
width, height = 1200, 630
bg_color = (0, 0, 0)        # black background
img = Image.new("RGB", (width, height), bg_color)
draw = ImageDraw.Draw(img)

# Try to use a nice bold font (falls back gracefully)
try:
    # Try Google Fonts-style bold first
    font = ImageFont.truetype("arial.ttf", 480)
except:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 480)  # macOS
    except:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 480)  # Linux
        except:
            font = ImageFont.load_default()  # last resort

# Ω symbol centered with glowing orange gradient effect
center_x, center_y = width // 2, height // 2 + 30

# Outer glow (multiple layers)
for glow in range(60, 0, -6):
    glow_color = (255, 147 + glow//3, 26 + glow//4)  # orange → bright
    draw.text(
        (center_x - 2, center_y - 2),
        "Ω",
        font=font,
        fill=glow_color,
        anchor="mm",
        stroke_width=4,
        stroke_fill=glow_color
    )

# Core bright symbol
draw.text(
    (center_x, center_y),
    "Ω",
    font=font,
    fill=(255, 147, 26),      # #f7931a Bitcoin orange
    anchor="mm",
    stroke_width=8,
    stroke_fill=(255, 180, 80)
)

# Optional tiny text at bottom
small_font = ImageFont.truetype("arial.ttf", 36) if font != ImageFont.load_default() else ImageFont.load_default()
draw.text(
    (center_x, height - 80),
    "Ωmega Pruner v10.0 — NUCLEAR EDITION",
    font=small_font,
    fill=(180, 180, 180),
    anchor="mm"
)

# Save
output_path = "omega_thumbnail.png"
img.save(output_path, "PNG")
print(f"Omega thumbnail generated → {os.path.abspath(output_path)}")
print("Now commit and push this file to your repo!")
