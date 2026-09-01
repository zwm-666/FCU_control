"""Word-safe SVG figure toolkit for the thesis.

WHY THIS MODULE EXISTS
  Microsoft Word ships a deliberately limited SVG renderer. Matplotlib's SVG
  output, at its defaults, uses three constructs that are exactly where Word is
  weakest, so figures can open with missing glyphs, shifted labels, or blank
  panels:

    1. <style> CSS blocks + class="..." attributes.
       Word's renderer does not apply CSS reliably; presentation must be on
       each element as attributes.
    2. <use xlink:href="#glyph-id"> symbol references for text and markers.
       Word frequently drops xlink-referenced content. Verified on this
       machine: a default 4x3 matplotlib figure emitted 82 xlink:href uses.
    3. Live <text> elements depending on installed fonts.
       Word does NOT honour fonts embedded in an SVG (it also cannot see Office
       cloud fonts such as Aptos from other applications), so a Chinese label
       in SimSun renders as boxes/tofu wherever that font is absent.

  This module removes all three:
    * svg.fonttype = "path" turns every glyph into a filled vector outline, so
      NO font is required at open time and CJK text can never fall back to
      tofu. Trade-off, stated plainly: outlined text is no longer selectable or
      searchable inside Word. For thesis figures that is the right trade,
      because a missing glyph is fatal and a non-searchable axis label is not.
      Axis/legend text stays real text ONLY if opted in via text_as_text=True.
    * inline_svg_styles() expands every <style> rule onto its elements and
      deletes the stylesheet.
    * flatten_svg_uses() replaces each <use> with a real copy of its target,
      then drops the now-unreferenced <defs>, so nothing depends on xlink.

  Everything else is deliberately conservative: no gradients, no filters, no
  transparency on fills, no clip-paths beyond the axes rectangle, no embedded
  raster. Line styles and markers, not colour alone, distinguish series, so the
  figures survive a greyscale print of the bound thesis.

VERIFY, DO NOT ASSUME
  validate_word_svg() re-parses each written file and fails loudly if any
  Word-hostile construct survived. Every public plotting helper calls it.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# ---------------------------------------------------------------------------
# Style: journal-plain, greyscale-safe, CJK-capable
# ---------------------------------------------------------------------------
# Serif Latin + SimSun CJK matches a Chinese thesis body text convention.
CJK_FALLBACK = ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Sans"]

# Colour-blind-safe (Okabe-Ito derived), and each series also gets a distinct
# dash pattern and marker so the figure reads correctly in black and white.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#7F7F7F", "#333333"]
DASHES = [(None, None), (5, 2), (2, 2), (7, 2, 1, 2),
          (4, 1, 1, 1), (9, 3), (1, 1.6), (6, 2, 2, 2)]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def apply_thesis_style(base_font_pt: float = 9.0) -> None:
    """Set rcParams for Word-safe, print-safe vector output."""
    plt.rcParams.update({
        # text -> vector outlines: no font needed by the viewer, no tofu
        "svg.fonttype": "path",
        "svg.hashsalt": "thesis",          # deterministic ids across runs
        "font.family": CJK_FALLBACK,
        "axes.unicode_minus": False,        # real hyphen, not U+2212
        "font.size": base_font_pt,
        "axes.titlesize": base_font_pt + 1,
        "axes.labelsize": base_font_pt,
        "xtick.labelsize": base_font_pt - 1,
        "ytick.labelsize": base_font_pt - 1,
        "legend.fontsize": base_font_pt - 1,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.linestyle": "-",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "legend.frameon": True,
        "legend.framealpha": 1.0,           # opaque: Word alpha is unreliable
        "legend.edgecolor": "#666666",
        "legend.fancybox": False,
        "patch.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,       # never transparent for Word
        "path.simplify": True,
        "path.simplify_threshold": 0.111,
    })


# ---------------------------------------------------------------------------
# SVG post-processing
# ---------------------------------------------------------------------------
def _parse_style_block(css: str) -> Dict[str, str]:
    """Map a simple '.cls {a:b;}' stylesheet to {'cls': 'a:b;'}."""
    out: Dict[str, str] = {}
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selectors, body = match.group(1), match.group(2).strip()
        for sel in selectors.split(","):
            sel = sel.strip()
            if sel.startswith("*."):
                sel = sel[2:]
            elif sel.startswith("."):
                sel = sel[1:]
            elif " ." in sel:
                sel = sel.split(" .")[-1]
            if sel:
                out[sel] = body
    return out


def inline_svg_styles(root: ET.Element) -> int:
    """Push CSS declarations onto elements as attributes; drop <style>."""
    styles: Dict[str, str] = {}
    style_nodes: List[Tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == f"{{{SVG_NS}}}style":
                styles.update(_parse_style_block(child.text or ""))
                style_nodes.append((parent, child))
    if not styles:
        for parent, node in style_nodes:
            parent.remove(node)
        return 0

    touched = 0
    for element in root.iter():
        cls = element.get("class")
        if not cls:
            continue
        merged: Dict[str, str] = {}
        for token in cls.split():
            for decl in styles.get(token, "").split(";"):
                if ":" in decl:
                    prop, value = decl.split(":", 1)
                    merged[prop.strip()] = value.strip()
        for prop, value in merged.items():
            # do not overwrite an explicit presentation attribute
            if element.get(prop) is None:
                element.set(prop, value)
        del element.attrib["class"]
        touched += 1
    for parent, node in style_nodes:
        parent.remove(node)
    return touched


def flatten_svg_uses(root: ET.Element) -> int:
    """Replace every <use> with a deep copy of its referenced node."""
    import copy

    def index_ids(node: ET.Element, table: Dict[str, ET.Element]) -> None:
        for child in node.iter():
            ident = child.get("id")
            if ident:
                table[ident] = child

    ids: Dict[str, ET.Element] = {}
    index_ids(root, ids)

    replaced = 0
    for _ in range(12):                     # <use> may reference a <use>
        pending: List[Tuple[ET.Element, int, ET.Element]] = []
        for parent in root.iter():
            for position, child in enumerate(list(parent)):
                if child.tag != f"{{{SVG_NS}}}use":
                    continue
                href = (child.get(f"{{{XLINK_NS}}}href")
                        or child.get("href") or "")
                target = ids.get(href.lstrip("#"))
                if target is None:
                    continue
                pending.append((parent, position, child))
        if not pending:
            break
        for parent, position, child in pending:
            href = (child.get(f"{{{XLINK_NS}}}href")
                    or child.get("href") or "")
            target = ids.get(href.lstrip("#"))
            if target is None:
                continue
            clone = copy.deepcopy(target)
            clone.attrib.pop("id", None)
            group = ET.Element(f"{{{SVG_NS}}}g")
            # <use x/y> is equivalent to a translate on the copied content
            dx, dy = child.get("x"), child.get("y")
            if dx or dy:
                group.set("transform",
                          f"translate({float(dx or 0)} {float(dy or 0)})")
            for prop, value in child.attrib.items():
                if prop in {"x", "y", "id", "width", "height",
                            f"{{{XLINK_NS}}}href", "href"}:
                    continue
                group.set(prop, value)
            group.append(clone)
            parent.remove(child)
            parent.insert(position, group)
            replaced += 1

    # remove <defs> that nothing references any more
    remaining = set()
    for element in root.iter():
        href = (element.get(f"{{{XLINK_NS}}}href")
                or element.get("href") or "")
        if href.startswith("#"):
            remaining.add(href[1:])
        cp = element.get("clip-path") or ""
        found = re.search(r"url\(#([^)]+)\)", cp)
        if found:
            remaining.add(found.group(1))
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag != f"{{{SVG_NS}}}defs":
                continue
            for node in list(child):
                if node.get("id") not in remaining:
                    child.remove(node)
            if len(child) == 0:
                parent.remove(child)
    return replaced


def _ensure_pixel_size(root: ET.Element) -> None:
    """Give the root explicit px width/height plus a viewBox.

    Word sizes an SVG from width/height; pt units with a decimal tail have been
    a source of mis-scaling, so convert to integral px (1pt = 4/3 px).
    """
    width, height = root.get("width", ""), root.get("height", "")

    def to_px(value: str) -> Optional[float]:
        match = re.match(r"^([0-9.]+)(pt|px|in|mm)?$", value.strip())
        if not match:
            return None
        number = float(match.group(1))
        unit = match.group(2) or "px"
        return {"pt": number * 4.0 / 3.0, "px": number,
                "in": number * 96.0, "mm": number * 96.0 / 25.4}[unit]

    w_px, h_px = to_px(width), to_px(height)
    if root.get("viewBox") is None and w_px and h_px:
        root.set("viewBox", f"0 0 {w_px:.2f} {h_px:.2f}")
    if w_px:
        root.set("width", f"{w_px:.2f}px")
    if h_px:
        root.set("height", f"{h_px:.2f}px")


def harden_svg_for_word(path: Path) -> Dict[str, int]:
    """Rewrite an SVG in place so Word renders it exactly as matplotlib did."""
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree = ET.parse(path)
    root = tree.getroot()

    # strip RDF/Dublin-Core metadata: pure noise for Word, and it carries a
    # creation timestamp that would make byte-identical reruns impossible
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in {f"{{{SVG_NS}}}metadata", f"{{{SVG_NS}}}title",
                             f"{{{SVG_NS}}}desc"}:
                parent.remove(child)

    inlined = inline_svg_styles(root)
    flattened = flatten_svg_uses(root)
    _ensure_pixel_size(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return {"styles_inlined": inlined, "uses_flattened": flattened}


def validate_word_svg(path: Path, allow_text: bool = False) -> Dict[str, object]:
    """Fail loudly if anything Word struggles with survived in the file."""
    raw = Path(path).read_text(encoding="utf-8")
    tree = ET.parse(path)
    root = tree.getroot()
    tags = {element.tag.split("}")[-1] for element in root.iter()}

    problems: List[str] = []
    if "style" in tags:
        problems.append("<style> CSS block present (Word may ignore it)")
    if "use" in tags:
        problems.append("<use> reference present (Word may drop it)")
    if "xlink:href" in raw:
        problems.append("xlink:href present")
    if 'class="' in raw:
        problems.append("class= attribute present (needs inline attributes)")
    for hostile in ("filter", "mask", "foreignObject", "switch", "image",
                    "linearGradient", "radialGradient", "pattern",
                    "animate", "script"):
        if hostile in tags:
            problems.append(f"<{hostile}> present (unsupported by Word)")
    if not allow_text and "text" in tags:
        problems.append("<text> present but text_as_text=False was requested")
    if root.get("width") is None or root.get("height") is None:
        problems.append("root <svg> lacks explicit width/height")
    if root.get("viewBox") is None:
        problems.append("root <svg> lacks viewBox (Word cannot scale it)")

    if problems:
        raise AssertionError(
            f"{path.name} is not Word-safe:\n  - " + "\n  - ".join(problems))

    return {
        "file": str(path),
        "bytes": len(raw.encode("utf-8")),
        "tags": sorted(tags),
        "paths": sum(1 for e in root.iter() if e.tag.endswith("}path")),
        "text_elements": sum(1 for e in root.iter() if e.tag.endswith("}text")),
        "width": root.get("width"),
        "height": root.get("height"),
        "viewBox": root.get("viewBox"),
    }


class _GlyphWarningTrap(logging.Handler):
    """Capture matplotlib's missing-glyph messages.

    matplotlib reports a missing glyph through the LOGGING system
    (matplotlib.font_manager / mathtext, level WARNING), not through
    warnings.warn, so warnings.catch_warnings never sees it. Attaching a
    handler to the 'matplotlib' logger is the only reliable interception point.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:
            return
        if "does not have a glyph" in text or "substituting with a dummy" in text:
            self.messages.append(text)


def save_word_svg(fig, path: Path, text_as_text: bool = False,
                  also_png: bool = True, png_dpi: int = 600) -> Dict[str, object]:
    """Save a figure as a hardened, validated SVG (+ optional PNG fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = plt.rcParams["svg.fonttype"]
    if text_as_text:
        plt.rcParams["svg.fonttype"] = "none"
    trap = _GlyphWarningTrap()
    mpl_logger = logging.getLogger("matplotlib")
    previous_level = mpl_logger.level
    previous_propagate = mpl_logger.propagate
    mpl_logger.addHandler(trap)
    mpl_logger.setLevel(logging.WARNING)
    try:
        # A missing glyph is the exact failure this module exists to prevent, and
        # matplotlib only *logs a warning* before substituting a dummy box
        # (tofu). Promote it to an error so it cannot reach the thesis unseen.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.02)
        missing = [str(w.message) for w in caught
                   if "does not have a glyph" in str(w.message)]
        missing.extend(trap.messages)
        if missing:
            unique = sorted(set(missing))
            raise AssertionError(
                f"{path.name}: {len(unique)} character(s) would render as tofu "
                f"boxes.\n  - " + "\n  - ".join(unique[:12]) +
                ("\n  ... " + str(len(unique) - 12) + " more" if len(unique) > 12 else "") +
                "\nCause is usually mathtext ($...$) forcing a math font with no "
                "CJK coverage, or a font family that lacks the character. Use "
                "plain unicode text, or add a covering font to CJK_FALLBACK."
            )
    finally:
        mpl_logger.removeHandler(trap)
        mpl_logger.setLevel(previous_level)
        mpl_logger.propagate = previous_propagate
        plt.rcParams["svg.fonttype"] = previous
    stats = harden_svg_for_word(path)
    report = validate_word_svg(path, allow_text=text_as_text)
    report.update(stats)
    if also_png:
        # A 600 dpi PNG twin is insurance: if a Word build still mis-renders the
        # vector, the figure can be swapped without regenerating anything.
        png_path = path.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=png_dpi,
                    bbox_inches="tight", pad_inches=0.02)
        report["png_twin"] = str(png_path)
    plt.close(fig)
    return report


# ---------------------------------------------------------------------------
# Sizing helpers: thesis text width is ~15.5 cm on A4 with 3 cm margins
# ---------------------------------------------------------------------------
CM = 1.0 / 2.54
FULL_W = 15.5 * CM       # single-column full width
HALF_W = 7.5 * CM        # side-by-side


def new_fig(width_cm: float = 15.5, height_cm: float = 7.0, **kwargs):
    return plt.subplots(figsize=(width_cm * CM, height_cm * CM), **kwargs)


def style_series(index: int) -> Dict[str, object]:
    """Colour + dash + marker for series `index`, greyscale-distinguishable."""
    dash = DASHES[index % len(DASHES)]
    style: Dict[str, object] = {
        "color": PALETTE[index % len(PALETTE)],
        "marker": MARKERS[index % len(MARKERS)],
    }
    if dash[0] is None:
        style["linestyle"] = "-"
    else:
        style["linestyle"] = (0, dash)
    return style


def hatch_for(index: int) -> str:
    """Bar hatch patterns, so bars remain distinct in a greyscale print."""
    return ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo", "***"][index % 8]


def vector_heatmap(ax, matrix, cmap="YlGnBu", vmin=None, vmax=None):
    """Draw a heatmap as real vector rectangles instead of a raster image.

    WHY: ax.imshow() rasterises the grid and embeds it as a base64
    <image xlink:href="data:image/png;base64,...">. Word cannot be relied on to
    render xlink-referenced embedded images, and it also defeats the whole point
    of a vector figure (the cells blur when the thesis is zoomed or printed).
    pcolormesh with snap/edges emits one <path> per cell, which Word draws
    natively and which stays sharp at any zoom.

    Returns the QuadMesh so a colorbar can still be attached.
    """
    import matplotlib as mpl

    data = np.asarray(matrix, dtype=float)
    rows, cols = data.shape
    vmin = float(np.nanmin(data)) if vmin is None else float(vmin)
    vmax = float(np.nanmax(data)) if vmax is None else float(vmax)
    # Cell edges chosen so cell (r, c) is centred on integer (c, r), matching
    # imshow's coordinate convention and therefore all existing tick logic.
    x_edges = np.arange(cols + 1) - 0.5
    y_edges = np.arange(rows + 1) - 0.5
    mesh = ax.pcolormesh(
        x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="white", linewidth=0.4, shading="flat", snap=True,
        rasterized=False,
    )
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[-1], y_edges[0])   # top row first, like imshow
    ax.set_aspect("auto")
    return mesh


def vector_colorbar(fig, ax, cmap, vmin, vmax, label="", n_steps=64,
                    width=0.018, pad=0.012, n_ticks=5, fontsize=6.6):
    """Draw a colourbar out of stacked vector rectangles.

    WHY: fig.colorbar() draws the gradient strip with an AxesImage, which the
    SVG backend emits as a base64 <image xlink:href="data:image/png;...">.
    That is the same Word-hostile construct vector_heatmap() exists to avoid,
    so the colourbar has to be built from real rectangles too.

    Returns the colourbar axes.
    """
    import matplotlib as mpl

    box = ax.get_position()
    cax = fig.add_axes([box.x1 + pad, box.y0, width, box.height])
    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap

    edges = np.linspace(0.0, 1.0, int(n_steps) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        cax.add_patch(mpl.patches.Rectangle(
            (0.0, lo), 1.0, hi - lo,
            facecolor=colormap(0.5 * (lo + hi)), edgecolor="none",
            linewidth=0.0,
        ))
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.set_xticks([])
    ticks = np.linspace(0.0, 1.0, int(n_ticks))
    cax.set_yticks(ticks)
    cax.set_yticklabels([f"{vmin + t * (vmax - vmin):.0f}" for t in ticks],
                        fontsize=fontsize)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    if label:
        cax.set_ylabel(label, fontsize=fontsize + 0.4)
    for spine in cax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("black")
    cax.grid(False)
    cax.tick_params(length=2.0, width=0.6)
    return cax
