"""Side-by-side "reference photo -> LEGO build" images for the README.

    uv run --with moderngl python scripts/showcase/compare.py

Reference photos are read from scripts/showcase/refs/ (not committed; bring your own).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from glrender import REPO, Renderer, load_model, mesh_for

HERE = Path(__file__).parent
REFS = HERE / "refs"
OUT = REPO / "docs"
FONT = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/noto/NotoSans-Regular.ttf"

# name, reference photo, camera (az, el, zoom, target or None), caption
SHOTS = [
    ("airliner_cabin", "airliner.png", (-3, 13, 1.45, (160, 110, -260)), "1950s airliner cabin"),
    ("bowling_alley", "bowling.png", (58, 38, 1.2, None), "Retro bowling alley"),
    ("helms_deep", "helms_film.png", (-28, 36, 2.1, (2400, 380, -1700)), "Helm's Deep (the Hornburg)"),
]


def font(size, bold=True):
    try:
        return ImageFont.truetype(FONT if bold else FONT_REG, size)
    except OSError:
        return ImageFont.load_default(size)


def fit(img, w, h):
    img = img.copy()
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    return img


def compare(name, ref, cam, caption, H=560):
    photo = Image.open(REFS / ref).convert("RGB")
    pw = round(photo.width * H / photo.height)
    photo = photo.resize((pw, H), Image.Resampling.LANCZOS)
    rw = max(pw, round(H * 1.5))
    r = Renderer(mesh_for(name), rw, H, background=((244, 246, 249), (214, 219, 227)))
    az, el, zoom, target = cam
    shot = r.frame(az, el, zoom=zoom, target=target)
    gap, pad, head = 24, 28, 64
    W = pad * 2 + pw + gap + 70 + rw
    sheet = Image.new("RGB", (W, H + head + pad * 2), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    x0, y0 = pad, pad + head
    sheet.paste(photo, (x0, y0))
    ax = x0 + pw + gap
    d.polygon([(ax, y0 + H / 2 - 22), (ax + 46, y0 + H / 2), (ax, y0 + H / 2 + 22)], fill=(242, 194, 0))
    sheet.paste(shot, (ax + 70, y0))
    n = len(load_model(name).parts)
    d.text((x0, pad + 8), "Reference photo", font=font(30), fill=(30, 32, 38))
    d.text((ax + 70, pad + 8), f"{caption} — {n:,} parts", font=font(30), fill=(30, 32, 38))
    out = OUT / f"compare_{name}.jpg"
    sheet.save(out, quality=88, optimize=True)
    print("wrote", out, sheet.size)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for shot in SHOTS:
        compare(*shot)
