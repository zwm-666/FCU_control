"""Build a real .docx that embeds the thesis figures the way Word itself does.

WHY THIS EXISTS
  Claiming "these SVGs work in Word" is not verification. This script produces
  an actual .docx containing every figure, so the claim can be checked by
  opening one file. It also IS the deliverable: a ready-to-use figure appendix
  the thesis can copy from.

HOW WORD STORES AN SVG (and why python-docx cannot do it)
  python-docx raises UnrecognizedImageError on .svg -- it has no SVG handler.
  Word 2016+ does support SVG, but it does not store it as a bare image. It
  writes BOTH:
     word/media/imageN.png   <- raster fallback, referenced by <a:blip r:embed>
     word/media/imageN.svg   <- the vector, referenced by an <a:extLst> child
                                <asvg:svgBlip r:embed> under the extension URI
                                {96DAC541-7B7A-43D3-8B79-37D633B846F1}
  Modern Word renders the SVG; anything older (or Google Docs, WPS, macOS
  Preview, PDF exporters that ignore the extension) silently falls back to the
  PNG. That dual form is strictly safer than either alone, which is why this
  script reproduces it rather than inserting only one format.

  We build the drawing XML directly: python-docx creates the paragraph and the
  relationships, then we swap in the blip structure. Nothing is hand-edited as
  text inside the zip -- all mutations go through python-docx/lxml.

VERIFICATION
  verify_docx() reopens the written file, then checks: the zip is intact, every
  figure contributed both an .svg and a .png part, every relationship target
  resolves, and each drawing carries a real svgBlip pointing at an SVG part.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import nsmap, qn
from docx.shared import Mm, Pt

FIG_DIR = Path(r"D:/learn/毕业材料/graduation/results/figures")
OUT_DOCX = Path(r"D:/learn/毕业材料/graduation/results/论文插图_Word兼容性验证.docx")

SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
ASVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
SVG_CONTENT_TYPE = "image/svg+xml"

# Figure order and the caption each one gets in the document.
FIGURES: List[tuple[str, str]] = [
    ("图1_主对比_10模型准确率与宏F1",
     "10种方法在物理判据数据集上的测试集性能（5种子均值±标准差）"),
    ("图2_噪声鲁棒性_准确率随SNR变化",
     "噪声鲁棒性：准确率随信噪比变化（仅测试输入加噪）"),
    ("图3_噪声退化幅度", "相对无噪工况的准确率下降幅度"),
    ("图11_噪声鲁棒性热力图", "噪声鲁棒性热力图（模型×信噪比）"),
    ("图4_消融实验_组件贡献", "消融实验：逐组件移除对准确率的影响（干净条件）"),
    ("图13_消融三条件对照_DI-EMSTGAT",
     "DI-EMSTGAT 消融三条件对照：条件间符号一致才可信"),
    ("图13_消融三条件对照_AB-EMSTGAT",
     "AB-EMSTGAT 消融三条件对照：全部变体方向不一致"),
    ("图5_消融_参数代价与精度收益", "各组件的参数代价与精度收益"),
    ("图6_最优模型混淆矩阵", "最优模型的测试集混淆矩阵"),
    ("图7_类别级F1对比", "各方法的类别级F1对比"),
    ("图12_逐种子准确率分布", "逐种子准确率分布"),
    ("图8_按行与按块划分的诚实基线差异",
     "按行随机划分与按录制段分组划分的诚实基线差异"),
    ("图9_数据集构成对照", "数据集构成对照：真实/合成/重复行"),
    ("图10_水活度物理判据", "第一性原理阴极水平衡判据的类间分离"),
]


def _register_asvg() -> None:
    """Make the 'asvg' prefix available to python-docx's qn() helper."""
    nsmap.setdefault("asvg", ASVG_NS)


def add_svg_picture(document: Document, paragraph, svg_path: Path,
                    png_path: Path, width_mm: float = 155.0) -> Dict[str, str]:
    """Insert an SVG with a PNG fallback, exactly as Word structures it."""
    part = document.part

    # PNG first: python-docx understands it, so it builds the whole drawing
    # (extents, docPr, blipFill) for us with correct EMU sizing.
    run = paragraph.add_run()
    shape = run.add_picture(str(png_path), width=Mm(width_mm))

    # Register the SVG as an additional image part and relate it to this part.
    svg_bytes = svg_path.read_bytes()
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI

    existing = [p for p in part.package.iter_parts()
                if str(p.partname).startswith("/word/media/")]
    index = len(existing) + 1
    svg_partname = PackURI(f"/word/media/{svg_path.stem}_{index}.svg")
    svg_part = Part(svg_partname, SVG_CONTENT_TYPE, svg_bytes, part.package)
    svg_rid = part.relate_to(svg_part, RT.IMAGE)

    # Attach <a:extLst><a:ext uri=...><asvg:svgBlip r:embed="rIdN"/>
    blip = shape._inline.graphic.graphicData.pic.blipFill.blip
    ext_lst = blip.makeelement(qn("a:extLst"), {})
    ext = blip.makeelement(qn("a:ext"), {"uri": SVG_EXT_URI})
    svg_blip = blip.makeelement(qn("asvg:svgBlip"), {qn("r:embed"): svg_rid})
    ext.append(svg_blip)
    ext_lst.append(ext)
    blip.append(ext_lst)

    return {"svg_part": str(svg_partname), "svg_rid": svg_rid,
            "png_rid": blip.get(qn("r:embed"))}


def build() -> Dict[str, object]:
    _register_asvg()
    document = Document()

    # A4 with the thesis margins, so the 155 mm figures visibly fit.
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Mm(30)
    section.right_margin = Mm(25)
    section.top_margin = Mm(25)
    section.bottom_margin = Mm(25)

    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    # East-Asian font must be set separately or Word substitutes for CJK runs.
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    document.add_heading("论文插图 — Word 兼容性验证文档", level=0)
    intro = document.add_paragraph()
    intro.add_run(
        "本文档由脚本自动生成，用于验证论文插图在 Microsoft Word 中的显示效果。"
        "每张图均按 Word 自身的方式嵌入：矢量 SVG 与 600 dpi PNG 同时存入，"
        "SVG 通过 svgBlip 扩展引用。较新版本 Word 渲染矢量图；旧版本、"
        "WPS、Google Docs 等自动回退到 PNG，不会出现空白。"
    )
    check = document.add_paragraph()
    check.add_run("请逐张确认：").bold = True
    for item in ("中文标签与图例正常显示，无 □ 方块；",
                 "图宽 155 mm，未超出页边距；",
                 "放大至 300% 后线条与文字仍然锐利；",
                 "黑白打印预览下各曲线/柱体仍可区分。"):
        document.add_paragraph(item, style="List Bullet")

    inserted: List[Dict[str, object]] = []
    missing: List[str] = []
    for order, (stem, caption) in enumerate(FIGURES, start=1):
        svg_path = FIG_DIR / f"{stem}.svg"
        png_path = FIG_DIR / f"{stem}.png"
        if not (svg_path.exists() and png_path.exists()):
            missing.append(stem)
            continue
        document.add_paragraph()
        holder = document.add_paragraph()
        holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
        info = add_svg_picture(document, holder, svg_path, png_path)
        cap = document.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"图 {order}  {caption}")
        run.font.size = Pt(9)
        run.font.name = "Times New Roman"
        cap.style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        inserted.append({"order": order, "stem": stem, **info,
                         "svg_bytes": svg_path.stat().st_size,
                         "png_bytes": png_path.stat().st_size})

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT_DOCX)
    return {"inserted": inserted, "missing": missing}


def verify_docx(path: Path, expected: int) -> Dict[str, object]:
    """Reopen the written file and prove the dual-format embedding survived."""
    problems: List[str] = []
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            problems.append(f"corrupt zip entry: {bad}")
        names = archive.namelist()
        svgs = [n for n in names if n.startswith("word/media/") and n.endswith(".svg")]
        pngs = [n for n in names if n.startswith("word/media/") and n.endswith(".png")]
        content_types = archive.read("[Content_Types].xml").decode("utf-8")
        document_xml = archive.read("word/document.xml").decode("utf-8")
        rels = archive.read("word/_rels/document.xml.rels").decode("utf-8")

        if SVG_CONTENT_TYPE not in content_types:
            problems.append("image/svg+xml missing from [Content_Types].xml")
        if len(svgs) != expected:
            problems.append(f"{len(svgs)} svg parts, expected {expected}")
        if len(pngs) != expected:
            problems.append(f"{len(pngs)} png fallbacks, expected {expected}")

        # Count real elements, not substring hits: 'svgBlip' also appears in the
        # namespace declaration, which would inflate a naive str.count().
        import xml.etree.ElementTree as ET
        doc_root = ET.fromstring(archive.read("word/document.xml"))
        svg_blip_nodes = [e for e in doc_root.iter()
                          if e.tag == f"{{{ASVG_NS}}}svgBlip"]
        svg_blips = len(svg_blip_nodes)
        if svg_blips != expected:
            problems.append(f"{svg_blips} svgBlip elements, expected {expected}")
        if SVG_EXT_URI not in document_xml:
            problems.append("SVG extension URI absent from document.xml")

        # each svgBlip must point at a relationship whose target is an SVG part
        import re
        rel_targets = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
        for node in svg_blip_nodes:
            rid = node.get(f"{{{nsmap['r']}}}embed")
            target = rel_targets.get(rid, "")
            if not target.endswith(".svg"):
                problems.append(f"svgBlip {rid} does not target an SVG part "
                                f"(got {target!r})")

        # every relationship target must exist in the package
        import re
        for rid, target in re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels):
            if target.startswith(("http://", "https://")):
                continue
            resolved = f"word/{target}".replace("word/../", "")
            if resolved not in names:
                problems.append(f"relationship {rid} -> missing {target}")

        # every r:embed referenced by a blip must be a declared relationship
        declared = set(re.findall(r'Id="([^"]+)"', rels))
        for rid in set(re.findall(r'r:embed="([^"]+)"', document_xml)):
            if rid not in declared:
                problems.append(f"r:embed {rid} has no relationship")

    if problems:
        raise AssertionError(f"{path.name} failed verification:\n  - "
                             + "\n  - ".join(problems))
    return {"file": str(path), "bytes": path.stat().st_size,
            "svg_parts": len(svgs), "png_fallbacks": len(pngs),
            "svg_blips": svg_blips}


def main() -> None:
    result = build()
    inserted = result["inserted"]
    if result["missing"]:
        print(f"!! missing figure files, skipped: {result['missing']}")
    report = verify_docx(OUT_DOCX, expected=len(inserted))
    print(f"built {OUT_DOCX}")
    print(f"  figures embedded : {len(inserted)}")
    print(f"  svg parts        : {report['svg_parts']}")
    print(f"  png fallbacks    : {report['png_fallbacks']}")
    print(f"  svgBlip elements : {report['svg_blips']}")
    print(f"  file size        : {report['bytes']/1e6:.2f} MB")
    print("  verification     : PASSED "
          "(zip intact, dual-format embedding, all relationships resolve)")

    (OUT_DOCX.parent / "Word嵌入验证报告.json").write_text(
        json.dumps({"docx": report, "figures": inserted,
                    "embedding": "dual: PNG via a:blip + SVG via asvg:svgBlip "
                                 f"under extension {SVG_EXT_URI}",
                    "rationale": "modern Word renders the SVG; older Word, WPS, "
                                 "Google Docs and some PDF exporters ignore the "
                                 "extension and use the PNG, so no viewer shows "
                                 "a blank frame"},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
