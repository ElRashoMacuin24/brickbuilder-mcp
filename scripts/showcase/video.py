"""Vertical (9:16) showcase video: photo -> LEGO build, for TikTok / Reels / Shorts.

    uv run --with moderngl python scripts/showcase/video.py [out.mp4]

Needs ffmpeg with libx264 and the reference photos in scripts/showcase/refs/.
No audio track: add a sound in the app.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from glrender import REPO, Renderer, load_model, mesh_for

HERE = Path(__file__).parent
REFS = HERE / "refs"
W, H, FPS = 1080, 1920, 30
BG_TOP, BG_BOT = (26, 29, 38), (10, 11, 15)
YELLOW = (242, 194, 0)
WHITE = (245, 246, 248)
GREY = (160, 166, 178)
FONT_BLACK = "/usr/share/fonts/noto/NotoSans-Black.ttf"
FONT_BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
REPO_URL = "github.com/ElRashoMacuin24/brickbuilder-mcp"
DARK_BG = (BG_TOP, BG_BOT)


def font(size, black=True):
    return ImageFont.truetype(FONT_BLACK if black else FONT_BOLD, size)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def background():
    g = np.linspace(0, 1, H)[:, None, None]
    arr = np.array(BG_TOP) * (1 - g) + np.array(BG_BOT) * g
    return Image.fromarray(arr.repeat(W, 1).astype(np.uint8))


BG = background()


def text(img, xy, s, size, fill=WHITE, black=True, anchor="mm", alpha=1.0, shadow=True):
    """Draw text with a soft shadow; alpha fades it in."""
    if alpha <= 0:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(size, black)
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((xy[0] + 3, xy[1] + 5), s, font=f, fill=(0, 0, 0, 170), anchor=anchor)
        layer = Image.alpha_composite(layer, sh.filter(ImageFilter.GaussianBlur(6)))
        d = ImageDraw.Draw(layer)
    d.text(xy, s, font=f, fill=fill + (255,), anchor=anchor)
    if alpha < 1:
        a = np.array(layer)
        a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
        layer = Image.fromarray(a)
    img.alpha_composite(layer) if img.mode == "RGBA" else img.paste(layer, (0, 0), layer)


def pill(img, xy, s, size=34, fg=(20, 20, 24), bg=YELLOW):
    d = ImageDraw.Draw(img)
    f = font(size)
    l, t, r, b = d.textbbox(xy, s, font=f, anchor="mm")
    d.rounded_rectangle((l - 22, t - 12, r + 22, b + 12), radius=(b - t) // 2 + 12, fill=bg)
    d.text(xy, s, font=f, fill=fg, anchor="mm")


def rounded(photo, radius=28):
    mask = Image.new("L", photo.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *photo.size), radius=radius, fill=255)
    out = Image.new("RGBA", photo.size)
    out.paste(photo, (0, 0), mask)
    return out


def photo_card(name, width=980, max_h=560):
    p = Image.open(REFS / name).convert("RGB")
    s = min(width / p.width, max_h / p.height)
    return rounded(p.resize((round(p.width * s), round(p.height * s)), Image.Resampling.LANCZOS))


class Video:
    def __init__(self, out):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        self.last = None
        self.count = 0

    def write(self, img, fade_from_last=0.0):
        img = img.convert("RGB")
        if fade_from_last > 0 and self.last is not None:
            img = Image.blend(img, self.last, fade_from_last)
        self.proc.stdin.write(img.tobytes())
        self.last, self.count = img, self.count + 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class Preview(Video):
    """Saves every n-th frame as PNG instead of encoding (for checking layout)."""

    def __init__(self, out_dir, every=45):
        self.dir, self.every, self.last, self.count = Path(out_dir), every, None, 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, img, fade_from_last=0.0):
        img = img.convert("RGB")
        if fade_from_last > 0 and self.last is not None:
            img = Image.blend(img, self.last, fade_from_last)
        if self.count % self.every == 0:
            img.resize((W // 2, H // 2)).save(self.dir / f"f{self.count:04d}.png")
        self.last, self.count = img, self.count + 1

    def close(self):
        pass


def xfade(i, n=8):
    """Blend weight of the previous segment's last frame for the first n frames."""
    return max(0.0, 1 - (i + 1) / n) if i < n else 0.0


# ---------------------------------------------------------------- segments

def seg_hook(v, hd):
    n = int(2.8 * FPS)
    for i in range(n):
        t = i / n
        img = hd.frame(-55 + 25 * t, 36, zoom=0.95 + 0.25 * t, target=(2250, 700, -1650)).convert("RGBA")
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(shade).rectangle((0, 0, W, 620), fill=(0, 0, 0, 120))
        img.alpha_composite(shade)
        text(img, (W // 2, 250), "I gave Claude", 92, alpha=ease(t * 4))
        text(img, (W // 2, 360), "a photo…", 92, fill=YELLOW, alpha=ease(t * 4))
        text(img, (W // 2, 500), "…and it built it in LEGO", 60, alpha=ease((t - 0.35) * 4))
        v.write(img, xfade(i) if v.last is not None else 0)


def seg_build(v, r, photo, title, subtitle, cam, secs_build=3.6, secs_orbit=2.6, extra_photos=()):
    """Photo on top, the model building itself underneath, then a short orbit."""
    az0, az1, el, zoom, target = cam
    nb, no = int(secs_build * FPS), int(secs_orbit * FPS)
    photos = [photo, *extra_photos]
    top = 150
    for i in range(nb + no):
        building = i < nb
        t = i / (nb + no)
        tb = ease(i / nb) if building else 1.0
        level = -1 + tb * (r.ymax + 2)
        az = az0 + (az1 - az0) * ease(t)
        shot = r.frame(az, el, zoom=zoom, level=level if building else None, target=target)
        img = BG.copy().convert("RGBA")
        card = photos[min(int(i / (nb + no) * len(photos)), len(photos) - 1)]
        ph = card.height
        text(img, (W // 2, top - 70), title, 64)
        pill(img, (W // 2 - card.width // 2 + 95, top + 36), "PHOTO", 30)
        img.alpha_composite(card, ((W - card.width) // 2, top + 70))
        y_arrow = top + 70 + ph + 40
        d = ImageDraw.Draw(img)
        d.polygon([(W // 2 - 34, y_arrow), (W // 2 + 34, y_arrow), (W // 2, y_arrow + 40)], fill=YELLOW)
        ry = y_arrow + 70
        img.paste(shot, (0, ry))
        pill(img, (140, ry + 40), "LEGO", 30)
        n_parts = int(r.parts * tb)
        text(img, (W // 2, H - 150), f"{n_parts:,} bricks", 58, fill=YELLOW)
        text(img, (W // 2, H - 80), subtitle, 36, fill=GREY, black=False)
        v.write(img, xfade(i))


def seg_flyover(v, hd, secs=6.0):
    n = int(secs * FPS)
    for i in range(n):
        t = ease(i / n)
        img = hd.frame(-70 + 85 * t, 44 - 12 * t, zoom=1.3 + 0.75 * t,
                       target=(2350 + 150 * t, 500 + 150 * t, -1650 - 100 * t)).convert("RGBA")
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(shade).rectangle((0, H - 420, W, H), fill=(0, 0, 0, 130))
        img.alpha_composite(shade)
        text(img, (W // 2, H - 300), "Helm's Deep", 96, alpha=ease(i / 20))
        text(img, (W // 2, H - 190), "41,864 bricks · minifig scale", 46, fill=YELLOW, alpha=ease((i - 10) / 20))
        text(img, (W // 2, H - 120), "Deeping Wall · Hornburg · tower · great hall", 34, fill=GREY, black=False,
             alpha=ease((i - 20) / 20))
        v.write(img, xfade(i))


def seg_outro(v, hd, secs=4.0):
    n = int(secs * FPS)
    for i in range(n):
        t = i / n
        img = hd.frame(40 + 25 * t, 30, zoom=1.2, target=(2300, 450, -1650)).convert("RGBA")
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 175))
        img.alpha_composite(shade)
        text(img, (W // 2, 700), "Claude + LEGO", 100, alpha=ease(t * 5))
        text(img, (W // 2, 830), "photo in, buildable model out", 50, fill=YELLOW, alpha=ease(t * 5 - 0.4))
        text(img, (W // 2, 1010), "Free & open source MCP server", 46, alpha=ease(t * 5 - 0.8))
        text(img, (W // 2, 1090), "exports to Mecabricks & BrickLink Studio", 40, fill=GREY, black=False,
             alpha=ease(t * 5 - 1.0))
        pill(img, (W // 2, 1260), REPO_URL, 34) if t > 0.3 else None
        v.write(img, xfade(i))


def main(out, preview=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    print("loading meshes…", flush=True)
    rend = {}
    for name in ("airliner_cabin", "bowling_alley", "helms_deep"):
        sq = Renderer(mesh_for(name), W, 830, background=DARK_BG)
        sq.parts = len(load_model(name).parts)
        rend[name] = sq
    hd_full = Renderer(mesh_for("helms_deep"), W, H, background=DARK_BG)
    v = Preview(preview) if preview else Video(out)
    seg_hook(v, hd_full)
    print("hook", v.count, flush=True)
    seg_build(v, rend["airliner_cabin"], photo_card("airliner.png"), "1950s airliner cabin",
              "every seat fits a minifig", (-40, -5, 22, 1.1, (160, 130, -240)))
    print("airliner", v.count, flush=True)
    seg_build(v, rend["bowling_alley"], photo_card("bowling.png"), "Retro bowling alley",
              "ball returns, racks, scoring tables", (70, 35, 38, 1.2, None))
    print("bowling", v.count, flush=True)
    seg_build(v, rend["helms_deep"], photo_card("helms_film.png", max_h=520), "Then I went bigger…",
              "built from 3 reference images", (-45, -20, 36, 1.25, None), secs_build=6.0, secs_orbit=1.2,
              extra_photos=(photo_card("helms_art.png", max_h=520), photo_card("helms_painting.png", max_h=520)))
    print("helms build", v.count, flush=True)
    seg_flyover(v, hd_full)
    seg_outro(v, hd_full)
    v.close()
    print(f"wrote {out} ({v.count / FPS:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prev = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--preview=")), None)
    main(Path(args[0]) if args else REPO / "exports" / "showcase" / "brickbuilder_tiktok.mp4", prev)
