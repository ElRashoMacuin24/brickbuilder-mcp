"""MCP server: build LEGO models brick by brick and export LDraw for Mecabricks."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
import warnings
from typing import Literal

from PIL import Image as PILImage

from mcp.server.mcpserver import Image, MCPServer
from pydantic import BaseModel, Field

from . import catalog, ldraw, packing, render
from .model import BuildError, Model, legacy_part_id, safe_name

HOME = Path(os.environ.get("BRICKBUILDER_HOME", ldraw.PROJECT_ROOT))
MODELS_DIR = HOME / "models"
EXPORTS_DIR = HOME / "exports"

INSTRUCTIONS = """\
Build LEGO models on a stud grid, preview them, and export LDraw (.ldr) files that
Mecabricks (mecabricks.com > Workshop > File > Import) and Studio can open.

Coordinates: x = studs to the right, z = studs toward the BACK (z=0 is the front row),
y = height in PLATES (brick = 3 plates, plate/tile = 1; y=0 is the ground).
A part's (x, y, z) is the min corner of its body after rotation. rot 0/90/180/270
(or front/right/back/left) turns parts about the vertical axis; for slopes it is the
direction the sloped face points (0 = toward the front). A 2x4 brick (3001) is 4 long
in x at rot 0 and 4 long in z at rot 90.

Workflow for building from a picture:
1. Look at the picture (Read the image file). Decide the scale (studs wide/tall/deep)
   and the main colours. image_to_grid helps: it downsamples a picture into rows of
   LEGO-colour letters.
2. new_model, then build bottom-up. fill_layers is the fastest path: give top-down
   colour maps per brick layer and it packs them into real bricks with staggered seams.
   Use add_parts for details: slopes for roofs and angled surfaces, tiles for smooth
   tops, plates for fine height steps.
3. render_model (optionally with reference_image) and compare against the picture;
   describe_model shows per-level maps. Iterate with remove_parts / add_parts / undo.
4. check_model to find floating or unconnected parts, then export_ldr.
"""

mcp = MCPServer("brickbuilder", instructions=INSTRUCTIONS)

# Limits that keep a single tool call from exhausting memory or CPU.
MAX_RENDER_SIZE = 1600
MAX_VIEWS = 9
MAX_GRID = 256
MAX_FILL_CELLS = 200_000
MAX_PARTS_PER_CALL = 5_000
# Refuse decompression bombs outright instead of just warning.
PILImage.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", PILImage.DecompressionBombWarning)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}


def _image_path(path: str) -> str:
    """Only picture files may be read by the image tools."""
    p = Path(path).expanduser()
    if p.suffix.lower() not in IMAGE_SUFFIXES:
        raise BuildError(f"Not an image file: {path} (expected one of {', '.join(sorted(IMAGE_SUFFIXES))})")
    if not p.is_file():
        raise BuildError(f"Image not found: {path}")
    return str(p)


def _check_grid(width: int, height: int | None) -> None:
    if not 1 <= width <= MAX_GRID or (height is not None and not 1 <= height <= MAX_GRID):
        raise BuildError(f"width and height must be between 1 and {MAX_GRID}")
_state: dict[str, Model] = {}


def _model() -> Model:
    if "current" not in _state:
        raise BuildError("No model open. Call new_model or open_model first.")
    return _state["current"]


def _save(m: Model) -> None:
    m.save(MODELS_DIR)


def _require_library() -> None:
    if not ldraw.library_available():
        raise BuildError(f"LDraw library not found at {ldraw.LDRAW_DIR}. Run scripts/fetch_ldraw.sh.")


class PartSpec(BaseModel):
    part: str = Field(description="LDraw part id, e.g. '3001' (brick 2x4)")
    color: int | str = Field(description="LDraw colour code or name, e.g. 4 or 'Red'")
    x: int
    y: int = Field(description="Bottom level in plates (brick = 3)")
    z: int
    rot: int | str = Field(0, description="0/90/180/270 or front/right/back/left")


class Region(BaseModel):
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int


# ---------------------------------------------------------------- models


@mcp.tool()
def new_model(name: str, description: str = "") -> str:
    """Start a new empty model (replaces the open one; models autosave)."""
    _require_library()
    _state["current"] = Model(name, description)
    _save(_state["current"])
    return f"Created model '{name}'. Saved automatically to {MODELS_DIR}/{safe_name(name)}.json"


@mcp.tool()
def open_model(name: str) -> str:
    """Open a previously saved model by name."""
    _require_library()
    path = MODELS_DIR / f"{safe_name(name)}.json"
    if not path.exists():
        raise BuildError(f"No saved model '{name}'. Available: {', '.join(list_models_raw()) or 'none'}")
    _state["current"] = Model.from_json(json.loads(path.read_text()))
    return f"Opened '{name}' with {len(_state['current'].parts)} parts."


def list_models_raw() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.json")) if MODELS_DIR.exists() else []


@mcp.tool()
def list_models() -> str:
    """List saved models."""
    return "\n".join(list_models_raw()) or "No saved models."


# ---------------------------------------------------------------- reference data


@mcp.tool()
def list_parts(category: str | None = None, search: str | None = None) -> str:
    """List the curated parts (brick, plate, tile, slope, round, misc) with their size,
    or search the whole LDraw library by words in the part title (e.g. 'window 1 x 2').
    Size is studs along x x studs along z at rot 0, and height in plates."""
    _require_library()
    lines = []
    if search:
        for pid, title in ldraw.search_library(search):
            try:
                sx, sz, h = catalog.footprint(pid)
                note = "" if pid in catalog.CATALOG else " (previewed as a box)"
                lines.append(f"{pid:10} {sx}x{sz} h{h}  {title}{note}")
            except ldraw.LDrawError:
                continue
        return "\n".join(lines) or "No matches."
    for pid, entry in catalog.CATALOG.items():
        if category and entry.category != category:
            continue
        sx, sz, h = catalog.footprint(pid)
        extra = ""
        if entry.shape.kind in ("slope", "slope_inv"):
            extra = " [slope face points front at rot 0]"
        lines.append(f"{entry.category:6} {pid:8} {sx}x{sz} h{h}  {ldraw.part_title(pid)}{extra}")
    return "\n".join(lines)


@mcp.tool()
def list_colors(search: str | None = None, include_special: bool = False) -> str:
    """List LDraw colours (code, name, hex). Solid colours only unless include_special."""
    _require_library()
    out = []
    for c in ldraw.colours().values():
        if c.code in (16, 24) or (not include_special and c.material != "solid"):
            continue
        if search and search.lower() not in c.name.lower():
            continue
        common = " *" if c.code in packing.STANDARD_PALETTE else ""
        out.append(f"{c.code:5} {c.hex} {c.name}{common}")
    return "\n".join(out) + "\n(* = common, widely available colour)"


# ---------------------------------------------------------------- building


@mcp.tool()
def add_parts(parts: list[PartSpec]) -> str:
    """Place one or more parts. Parts that collide or are invalid are skipped and reported;
    the rest are placed. Returns the new part ids."""
    m = _model()
    if len(parts) > MAX_PARTS_PER_CALL:
        raise BuildError(f"add_parts is limited to {MAX_PARTS_PER_CALL} parts per call")
    m.checkpoint()
    ok, errors = [], []
    for spec in parts:
        try:
            p = m.add(spec.part, spec.color, spec.x, spec.y, spec.z, spec.rot)
            ok.append(p.id)
        except (BuildError, ldraw.LDrawError) as e:
            errors.append(str(e))
    _save(m)
    msg = f"Placed {len(ok)} part(s): ids {_ranges(ok)}." if ok else "Placed nothing."
    if errors:
        msg += "\nErrors:\n- " + "\n- ".join(errors)
    return msg


@mcp.tool()
def remove_parts(ids: list[int] | None = None, region: Region | None = None) -> str:
    """Remove parts by id and/or every part touching an inclusive grid region."""
    m = _model()
    m.checkpoint()
    targets = set(ids or [])
    if region:
        targets |= set(m.parts_in_region(region.x0, region.x1, region.y0, region.y1, region.z0, region.z1))
    removed = m.remove(sorted(targets))
    _save(m)
    return f"Removed {len(removed)} part(s)."


@mcp.tool()
def fill_layers(layers: list[list[str]], legend: dict[str, int | str], x: int = 0, y: int = 0, z: int = 0,
                kind: Literal["brick", "plate", "tile"] = "brick", upright: bool = False) -> str:
    """Build volume from colour maps, automatically packed into bricks/plates/tiles.

    layers: bottom-to-top list of layers; each layer is a list of equal-length rows seen
    from ABOVE, first row = back, last row = front. Each character is a cell of 1 stud;
    '.' = empty. legend maps characters to colours, e.g. {"R": "Red", "W": 15}.
    Each layer is one brick (3 plates) or one plate tall depending on kind; layer i sits
    at y + i*height. (x, z) is the front-left corner of the map.

    upright=True: each layer is instead a front-facing wall picture (first row = top,
    last row = bottom), one course per row, at depth z + layer index. Handy for mosaics
    and facades.

    Cells already occupied are skipped. Seams are staggered between courses."""
    m = _model()
    cells = sum(len(row) for layer in layers for row in layer)
    if cells > MAX_FILL_CELLS:
        raise BuildError(f"fill_layers is limited to {MAX_FILL_CELLS} cells per call (got {cells}); split it up")
    m.checkpoint()
    try:
        res = packing.fill_layers(m, layers, legend, x, y, z, kind, upright)
    except (BuildError, ldraw.LDrawError):
        m.undo()
        raise
    _save(m)
    msg = f"Placed {len(res['placed'])} part(s): ids {_ranges(res['placed'])}."
    if res["skipped_occupied_cells"]:
        msg += f" Skipped {len(res['skipped_occupied_cells'])} already-occupied cells."
    return msg


@mcp.tool()
def image_to_grid(image_path: str, width: int, height: int | None = None, max_colors: int = 10,
                  palette: list[int | str] | None = None, crop: list[float] | None = None,
                  cell: Literal["stud", "brick", "plate"] = "stud") -> str:
    """Downsample a picture into rows of colour letters using LEGO colours, ready for
    fill_layers. crop = [left, top, right, bottom] fractions (0-1) to focus on the subject.
    cell sets the cell's aspect ratio when height is omitted: 'stud' for top-down maps,
    'brick' or 'plate' for upright walls. Transparent pixels become '.'."""
    _require_library()
    _check_grid(width, height)
    aspect = {"stud": 1.0, "brick": 1.2, "plate": 0.4}[cell]
    res = packing.image_to_grid(_image_path(image_path), width, height, max_colors, palette, crop, aspect)
    table = ldraw.colours()
    legend = "\n".join(f"  {ch} = {code} {table[code].name}" for ch, code in res["legend"].items())
    return (f"{res['width']}x{res['height']} grid (first row = top of image)\n" + "\n".join(res["rows"]) +
            f"\nlegend:\n{legend}\nlegend json: {json.dumps(res['legend'])}")


@mcp.tool()
def build_mosaic(image_path: str, width: int, orientation: Literal["upright", "flat"] = "upright",
                 height: int | None = None, max_colors: int = 10, crop: list[float] | None = None,
                 kind: Literal["brick", "plate"] = "plate", x: int = 0, y: int = 0, z: int = 0) -> str:
    """Build a picture directly as a mosaic in the current model.
    upright: a wall facing the front, 1 stud deep (kind=plate gives fine vertical detail,
    kind=brick is sturdier). flat: lying on the ground, one plate thick, image top = back."""
    m = _model()
    if orientation == "flat":
        kind = "plate"
    _check_grid(width, height)
    aspect = 1.0 if orientation == "flat" else (1.2 if kind == "brick" else 0.4)
    res = packing.image_to_grid(_image_path(image_path), width, height, max_colors, None, crop, aspect)
    m.checkpoint()
    try:
        out = packing.fill_layers(m, [res["rows"]], res["legend"], x, y, z, kind,
                                  upright=orientation == "upright")
    except (BuildError, ldraw.LDrawError):
        m.undo()
        raise
    _save(m)
    return f"Mosaic {res['width']}x{res['height']} built from {len(out['placed'])} parts, {len(res['legend'])} colours."


@mcp.tool()
def undo() -> str:
    """Undo the last building action."""
    m = _model()
    ok = m.undo()
    _save(m)
    return "Undone." if ok else "Nothing to undo."


# ---------------------------------------------------------------- inspection


@mcp.tool()
def describe_model(levels: list[int] | None = None, all_levels: bool = False, list_all_parts: bool = False) -> str:
    """Summarise the model: bounds, part counts, and top-down colour maps of plate levels
    (first row = back, last = front; uppercase letters = colours in the legend).
    Maps are shown for `levels`, or every level when all_levels is set."""
    m = _model()
    table = ldraw.colours()
    b = m.bounds()
    if b is None:
        return f"Model '{m.name}' is empty."
    x0, x1, y0, y1, z0, z1 = b
    out = [f"Model '{m.name}': {len(m.parts)} parts, bounds x {x0}..{x1}, y {y0}..{y1} (plates), z {z0}..{z1}",
           f"size: {x1 - x0 + 1} x {z1 - z0 + 1} studs, {(y1 - y0 + 1) / 3:.1f} bricks tall"]
    counts = Counter((p.part, p.color) for p in m.parts.values())
    out.append("parts:")
    for (pid, col), n in counts.most_common():
        out.append(f"  {n:4} x {pid:8} {table[col].name:24} {ldraw.part_title(pid)}")
    show = list(range(y0, y1 + 1)) if all_levels else (levels or [])
    legend: dict[int, str] = {}
    for yy in show:
        out.append(f"level y={yy} (map origin x={x0}, z={z0} at bottom-left):")
        out += ["  " + row for row in m.layer_map(yy, legend)]
    if legend:
        out.append("legend: " + ", ".join(f"{ch}={table[c].name}" for c, ch in legend.items()))
    if list_all_parts:
        out.append("placements (id part color x y z rot):")
        out += [f"  {p.id} {p.part} {p.color} {p.x} {p.y} {p.z} {p.rot}" for p in m.parts.values()]
    return "\n".join(out)


@mcp.tool()
def check_model() -> str:
    """Check structure: floating groups (not connected to anything touching the ground)
    and loose groups (standing on the ground but not attached to the main body)."""
    m = _model()
    r = m.check()
    lines = [f"{r['parts']} parts in {r['connected_groups']} stud-connected group(s); largest: {r['group_sizes']}"]
    for g in r["floating_groups"]:
        lines.append(f"FLOATING group of {len(g)} part(s): ids {_ranges(g)}")
    for g in r["loose_grounded_groups"]:
        lines.append(f"loose group on the ground ({len(g)} part(s)): ids {_ranges(g)}")
    if not r["floating_groups"] and not r["loose_grounded_groups"]:
        lines.append("OK: everything is connected.")
    return "\n".join(lines)


@mcp.tool()
def render_model(views: list[str] | None = None, size: int = 480, reference_image: str | None = None) -> Image:
    """Render a preview sheet. views: any of front, back, left, right, top, iso_front_left,
    iso_front_right, iso_back_left, iso_back_right, or 'azimuth,elevation' in degrees.
    Default: iso_front_left, iso_back_right, front, top. reference_image adds the source
    picture as the first panel for side-by-side comparison. Parts are simplified
    (round parts and non-catalog parts show as boxes)."""
    m = _model()
    views = (views or None) and views[:MAX_VIEWS]
    size = max(64, min(int(size), MAX_RENDER_SIZE))
    return Image(data=render.render(m, views, size, _image_path(reference_image) if reference_image else None),
                 format="png")


@mcp.tool()
def export_ldr(path: str | None = None, legacy_ids: bool = False) -> str:
    """Write the model as an LDraw .ldr file (one build step per level).
    Import it in Mecabricks via Workshop > File > Import. If some parts come in missing,
    re-export with legacy_ids=True (uses e.g. 3023 instead of 3023b)."""
    m = _model()
    out = Path(path).expanduser() if path else EXPORTS_DIR / f"{safe_name(m.name)}.ldr"
    if out.suffix.lower() not in (".ldr", ".mpd"):
        raise BuildError("export path must end in .ldr (or .mpd)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(m.to_ldr(legacy_ids))
    ids = sorted({legacy_part_id(p.part) if legacy_ids else p.part for p in m.parts.values()})
    return (f"Wrote {out} ({len(m.parts)} parts, {len(ids)} distinct part types: {', '.join(ids)}).\n"
            "Mecabricks: open the Workshop, File > Import, choose LDraw, pick this file.")


def _ranges(ids: list[int]) -> str:
    ids = sorted(ids)
    if not ids:
        return "none"
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(out)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
