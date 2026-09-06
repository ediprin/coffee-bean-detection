#!/usr/bin/env python3
"""Generate the 16-slide seminar-proposal deck from the approved Markdown source.

The supplied PPTX is the visual source of truth. The generator reuses its slide
master/layouts instead of cloning slide XML. This keeps the USU visual identity
while avoiding broken relationships and placeholder-order issues.

Usage:
    python tools/thesis_pptx/generate_seminar_proposal.py \
      --template assets/ppt/ppt_seminar_proposal_template.pptx \
      --source docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md \
      --output build/seminar_proposal_kopi.pptx
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

GREEN = RGBColor(77, 166, 62)
DARK_GREEN = RGBColor(45, 130, 44)
PALE_GREEN = RGBColor(227, 241, 217)
PALE_BLUE = RGBColor(202, 236, 246)
DARK = RGBColor(25, 25, 25)
WHITE = RGBColor(255, 255, 255)
GRAY = RGBColor(90, 90, 90)
EXPECTED_SLIDES = 16


def parse_source(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^---\s*$", text)
    slides = []
    for block in blocks:
        m = re.search(r"(?m)^## SLIDE\s+(\d+)\s+—\s+(.+?)\s*$", block)
        if not m:
            continue
        body = block[m.end():].strip()
        title_match = re.search(r"(?m)^###\s+(.+?)\s*$", body)
        title = title_match.group(1).strip() if title_match else ""
        if title_match:
            body = body[title_match.end():].strip()
        slides.append({
            "number": int(m.group(1)),
            "label": m.group(2).strip(),
            "title": title,
            "body": body,
        })
    return sorted(slides, key=lambda x: x["number"])


def delete_all_slides(prs: Presentation) -> None:
    while len(prs.slides):
        slide_id = prs.slides._sldIdLst[0]
        prs.part.drop_rel(slide_id.rId)
        del prs.slides._sldIdLst[0]


def blank_content_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    for shape in list(slide.shapes):
        if getattr(shape, "is_placeholder", False) and getattr(shape, "has_text_frame", False):
            shape.text_frame.clear()
    return slide


def add_text(slide, text, x, y, w, h, *, size=16, bold=False,
             color=DARK, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.06)
    tf.vertical_anchor = valign
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
    return box


def add_title(slide, text, size=22):
    return add_text(slide, text, 0.9, 0.75, 10.4, 0.55, size=size)


def add_panel(slide, x, y, w, h, *, fill=PALE_GREEN, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = fill
    return shape


def add_number_item(slide, number, text, y, *, x=0.95, text_w=4.0):
    tag = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(0.6), Inches(0.42)
    )
    tag.fill.solid(); tag.fill.fore_color.rgb = GREEN; tag.line.color.rgb = GREEN
    add_text(slide, f"{number:02d}", x + 0.07, y + 0.01, 0.46, 0.3,
             size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, text, x + 0.77, y - 0.02, text_w, 0.58, size=15)


def cover(prs, data):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            shape.text_frame.clear()
    title = slide.shapes.title
    title.text = data["title"]
    for p in title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = WHITE
    sub = slide.placeholders[1]
    sub.text = "[Nama Mahasiswa]\n[NIM]"
    for p in sub.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(12); r.font.color.rgb = WHITE
    add_text(slide, "Seminar Proposal", 2.1, 1.1, 3.2, 0.5,
             size=17, bold=True, color=WHITE)
    add_text(slide, "Dosen Pembimbing\n[Dosen Pembimbing 1]\n[Dosen Pembimbing 2]",
             7.55, 5.42, 4.6, 0.7, size=10, color=WHITE)


def agenda(prs, _):
    s = blank_content_slide(prs); add_title(s, "Pembahasan Materi", 24)
    add_number_item(s, 1, "Pendahuluan", 2.65, x=0.95, text_w=3.8)
    add_number_item(s, 2, "Penelitian Terdahulu", 2.65, x=6.65, text_w=4.0)
    add_number_item(s, 3, "Metodologi Penelitian", 4.05, x=0.95, text_w=4.4)


def intro(prs, _):
    s = blank_content_slide(prs); add_title(s, "Pendahuluan")
    pts = [
        "Inspeksi mutu biji kopi hijau masih banyak bergantung pada pengamatan visual, sehingga konsistensinya dapat dipengaruhi pengalaman dan kondisi pemeriksa.",
        "Deteksi otomatis menjadi lebih menantang ketika kategori cacat semakin rinci karena beberapa kelas memiliki kemiripan pada warna, tekstur, bentuk, dan detail lokal.",
        "Kondisi tersebut mendorong kebutuhan representasi citra yang lebih diskriminatif untuk deteksi fine-grained cacat biji kopi.",
    ]
    for i, (text, y) in enumerate(zip(pts, [2.0, 3.25, 4.5]), 1):
        add_number_item(s, i, text, y, text_w=9.4)


def problem(prs, _):
    s = blank_content_slide(prs); add_title(s, "Rumusan Masalah")
    add_panel(s, 0.45, 1.55, 11.9, 3.55, fill=RGBColor(180, 215, 168), radius=False)
    text = ("Deteksi fine-grained cacat biji kopi menghadapi kemiripan visual antarkelas, "
            "sedangkan pemanfaatan informasi frekuensi-angular sebelum proses deteksi masih terbatas. "
            "Penelitian ini mengkaji penerapan dan optimasinya pada YOLO26n terhadap kinerja deteksi "
            "dan biaya komputasi.")
    add_text(s, text, 0.9, 2.0, 10.9, 2.35, size=18,
             align=PP_ALIGN.JUSTIFY, valign=MSO_ANCHOR.MIDDLE)


def scope(prs, _):
    s = blank_content_slide(prs); add_title(s, "BATASAN MASALAH")
    add_panel(s, 0.35, 1.5, 12.0, 4.9, fill=RGBColor(221, 239, 204), radius=False)
    items = [
        "Deteksi fine-grained cacat biji kopi hijau.",
        "Dataset utama robusta_SNI_Dataset (21 kelas).",
        "Dataset Capstone, Lulus, dan Niacubilla sebagai konfirmasi.",
        "Model utama YOLO26n tanpa modifikasi backbone, neck, dan head.",
        "Optimasi difokuskan pada prapemrosesan frekuensi-angular.",
        "Evaluasi utama menggunakan mAP50–95 dan biaya komputasi end-to-end.",
    ]
    y = 1.85
    for item in items:
        add_text(s, "• " + item, 0.65, y, 11.3, 0.52, size=15); y += 0.7


def related(prs, _):
    s = blank_content_slide(prs); add_title(s, "Penelitian Terdahulu")
    rows = [
        ("Hong et al. (2026)", "Improved YOLOv10", "Deteksi cacat kopi"),
        ("Jiao et al. (2025)", "Multistage fusion + attention", "Diskriminasi fitur cacat kopi"),
        ("Li et al. (2025)", "Fourier preprocessing + YOLO", "Pemrosesan spektral sebelum deteksi"),
        ("Xu et al. (2025)", "AFAB", "Frekuensi-angular untuk fine-grained detection"),
    ]
    xs, widths, headers = [0.55, 3.2, 7.45], [2.6, 4.2, 4.55], ["Penelitian", "Pendekatan", "Fokus"]
    y, rh = 1.55, 0.72
    for x, w, h in zip(xs, widths, headers):
        add_panel(s, x, y, w, rh, fill=GREEN, radius=False)
        add_text(s, h, x + 0.06, y + 0.09, w - 0.12, 0.45, size=14,
                 bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    y += rh
    for r, row in enumerate(rows):
        fill = RGBColor(239, 246, 235) if r % 2 == 0 else RGBColor(224, 237, 218)
        for x, w, txt in zip(xs, widths, row):
            add_panel(s, x, y, w, rh, fill=fill, radius=False)
            add_text(s, txt, x + 0.08, y + 0.1, w - 0.16, 0.48,
                     size=11.5, align=PP_ALIGN.CENTER)
        y += rh
    add_panel(s, 0.55, 5.05, 11.45, 0.88, fill=RGBColor(217, 235, 203), radius=False)
    add_text(s, "Gap: penelitian pada kopi lebih banyak mengembangkan representasi internal model; "
                "prapemrosesan frekuensi-angular sebelum detektor belum dikaji secara khusus.",
             0.72, 5.18, 11.05, 0.58, size=12.5, bold=True)


def why(prs, _):
    s = blank_content_slide(prs); add_title(s, "Mengapa Frekuensi-Angular?")
    cards = [
        ("Fine-grained defect", "Perbedaan kecil pada tekstur dan pola permukaan"),
        ("Frequency", "Menangkap karakteristik perubahan dan detail visual"),
        ("Angular", "Menangkap distribusi respons berdasarkan arah"),
    ]
    for x, (head, body) in zip([0.7, 4.45, 8.2], cards):
        add_panel(s, x, 2.0, 3.15, 2.45)
        add_text(s, head, x + 0.15, 2.25, 2.85, 0.45, size=17,
                 bold=True, color=DARK_GREEN, align=PP_ALIGN.CENTER)
        add_text(s, body, x + 0.25, 3.0, 2.65, 1.0, size=14, align=PP_ALIGN.CENTER)
    add_text(s, "Hipotesis: representasi frekuensi-angular dapat membantu menghasilkan masukan yang "
                "lebih diskriminatif bagi detektor.", 1.0, 5.05, 10.8, 0.72,
             size=15, bold=True, align=PP_ALIGN.CENTER)


def dataset(prs, _):
    s = blank_content_slide(prs); add_title(s, "Dataset Penelitian")
    add_panel(s, 0.75, 1.7, 5.2, 3.45); add_panel(s, 6.4, 1.7, 5.2, 3.45)
    add_text(s, "Dataset Utama", 1.0, 2.0, 4.7, 0.45, size=18, bold=True,
             color=DARK_GREEN, align=PP_ALIGN.CENTER)
    add_text(s, "robusta_SNI_Dataset\n21 kelas\n\nPengembangan dan pemilihan C*",
             1.0, 2.65, 4.7, 1.75, size=16, align=PP_ALIGN.CENTER)
    add_text(s, "Dataset Konfirmasi", 6.65, 2.0, 4.7, 0.45, size=18, bold=True,
             color=DARK_GREEN, align=PP_ALIGN.CENTER)
    add_text(s, "Capstone — 14 kelas\nLulus — 6 kelas\nNiacubilla — 9 kelas",
             6.7, 2.75, 4.6, 1.55, size=16, align=PP_ALIGN.CENTER)
    add_text(s, "Split 70% train · 15% validation · 15% test  |  Setiap dataset digunakan secara terpisah.",
             1.2, 5.55, 10.4, 0.52, size=13, bold=True, align=PP_ALIGN.CENTER)


def methodology(prs, _):
    s = blank_content_slide(prs); add_title(s, "METODOLOGI PENELITIAN")
    add_text(s, "Empat kondisi eksperimen utama:", 0.9, 1.65, 5.0, 0.45, size=15)
    for x, (a, b) in zip([0.75, 3.75, 6.75, 9.75],
                         [("B0", "YOLO26n"), ("B1", "CLAHE → YOLO26n"),
                          ("B2", "C0 → YOLO26n"), ("B3", "C* → YOLO26n")]):
        add_panel(s, x, 2.25, 2.3, 1.15, fill=RGBColor(209, 232, 193))
        add_text(s, a, x + 0.08, 2.4, 0.55, 0.55, size=18, bold=True,
                 color=DARK_GREEN, align=PP_ALIGN.CENTER)
        add_text(s, b, x + 0.65, 2.42, 1.55, 0.55, size=13, align=PP_ALIGN.CENTER)
    add_panel(s, 0.9, 4.0, 11.2, 1.65, fill=RGBColor(221, 239, 204), radius=False)
    add_text(s, "B2 − B0  → efek frequency-angular reference frontend\n"
                "B3 − B2  → efek optimasi desain\n"
                "B3 − B1  → perbandingan terhadap CLAHE",
             1.25, 4.25, 10.5, 1.15, size=14, align=PP_ALIGN.CENTER)


def flow(prs, title, steps, footer):
    s = blank_content_slide(prs); add_title(s, title)
    y0, gap, avail = 1.6, 0.12, 4.7
    h = (avail - gap * (len(steps) - 1)) / len(steps)
    for i, step in enumerate(steps):
        y = y0 + i * (h + gap)
        add_panel(s, 3.05, y, 6.7, h)
        add_text(s, step, 3.25, y + 0.05, 6.3, h - 0.1, size=12.5,
                 bold=i in (0, len(steps) - 1), align=PP_ALIGN.CENTER,
                 valign=MSO_ANCHOR.MIDDLE)
    add_text(s, footer, 1.0, 6.15, 10.9, 0.4, size=12.5,
             bold=True, align=PP_ALIGN.CENTER)


def preprocessing(prs, _):
    flow(prs, "Alur Prapemrosesan Frekuensi-Angular",
         ["Citra RGB", "Patch Lokal", "FFT 2D", "Analisis Amplitudo & Arah",
          "Adaptive Spectral Weighting", "Inverse FFT", "Rekonstruksi",
          "Residual Fusion", "YOLO26n"],
         "I′ = I + I ⊙ G   ·   Parameter-free frontend   ·   YOLO26n tidak dimodifikasi")


def design(prs, _):
    flow(prs, "Optimasi Desain",
         ["C0 — Reference frequency-angular", "C1 — + Hann window",
          "C2 — + Unsigned orientation", "C3 — + Radial bands",
          "C4 — + Soft threshold", "C5 — + Luminance guidance",
          "C* — konfigurasi terpilih"],
         "Setiap konfigurasi menambahkan satu perubahan utama secara kumulatif.")


def research_flow(prs, _):
    flow(prs, "Alur Penelitian",
         ["robusta_SNI_Dataset", "Split 70 / 15 / 15", "Baseline B0",
          "Tetapkan Hard Classes", "Evaluasi C0–C5", "Sensitivity Analysis",
          "Pilih & Bekukan C*", "Multi-seed Confirmation", "Final Test"],
         "Test set tidak digunakan untuk memilih C*.")


def cross_dataset(prs, _):
    s = blank_content_slide(prs); add_title(s, "Konfirmasi Lintas Dataset")
    add_panel(s, 4.3, 1.6, 4.2, 0.8, fill=RGBColor(209, 232, 193))
    add_text(s, "C* dibekukan", 4.45, 1.78, 3.9, 0.45, size=18,
             bold=True, color=DARK_GREEN, align=PP_ALIGN.CENTER)
    for x, (name, nclass) in zip([1.0, 4.65, 8.3],
                                  [("Capstone", "14 kelas"), ("Lulus", "6 kelas"),
                                   ("Niacubilla", "9 kelas")]):
        add_panel(s, x, 3.0, 3.0, 1.7)
        add_text(s, name, x + 0.15, 3.25, 2.7, 0.45, size=17,
                 bold=True, align=PP_ALIGN.CENTER)
        add_text(s, nclass + "\nB0 vs B3", x + 0.15, 3.8, 2.7, 0.65,
                 size=14, align=PP_ALIGN.CENTER)
    add_text(s, "Seeds: 123 · 2026 · 31415", 2.0, 5.25, 8.8, 0.42,
             size=14, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, "Tidak ada retuning C* pada dataset konfirmasi.",
             2.0, 5.75, 8.8, 0.42, size=13, align=PP_ALIGN.CENTER)


def objective(prs, _):
    s = blank_content_slide(prs); add_title(s, "Tujuan Penelitian")
    add_panel(s, 0.8, 2.05, 11.0, 2.65, fill=PALE_BLUE, radius=False)
    add_text(s, "Menganalisis dan mengoptimasi prapemrosesan citra berbasis frekuensi-angular pada "
                "YOLO26n untuk deteksi fine-grained cacat biji kopi serta mengevaluasi pengaruhnya "
                "terhadap kinerja deteksi dan biaya komputasi.",
             1.25, 2.55, 10.1, 1.65, size=18,
             align=PP_ALIGN.JUSTIFY, valign=MSO_ANCHOR.MIDDLE)


def evaluation(prs, _):
    s = blank_content_slide(prs); add_title(s, "Evaluasi Penelitian")
    cols = [
        ("Deteksi", ["mAP50–95", "mAP50", "Precision & Recall"]),
        ("Fine-grained", ["AP per kelas", "AP_H", "AP_worst"]),
        ("Efisiensi", ["Preprocessing time", "End-to-end latency", "FPS", "Peak GPU memory"]),
    ]
    for x, (head, items) in zip([0.65, 4.45, 8.25], cols):
        add_panel(s, x, 1.8, 3.25, 3.95, fill=PALE_BLUE, radius=False)
        add_text(s, head, x + 0.15, 2.05, 2.95, 0.45, size=18,
                 bold=True, color=DARK_GREEN, align=PP_ALIGN.CENTER)
        y = 2.85
        for item in items:
            add_text(s, "• " + item, x + 0.35, y, 2.65, 0.45, size=14); y += 0.63


def closing(prs, _):
    s = blank_content_slide(prs)
    add_text(s, "TERIMA KASIH", 0.9, 2.7, 7.8, 1.0,
             size=32, bold=True, color=GREEN)
    add_text(s, "Pertanyaan & Diskusi", 0.95, 3.7, 5.2, 0.55,
             size=16, color=GRAY)


BUILDERS = {
    1: cover, 2: agenda, 3: intro, 4: problem, 5: scope, 6: related,
    7: why, 8: dataset, 9: methodology, 10: preprocessing, 11: design,
    12: research_flow, 13: cross_dataset, 14: objective, 15: evaluation,
    16: closing,
}


def build(template: Path, source: Path, output: Path) -> None:
    slides_data = parse_source(source)
    if len(slides_data) != EXPECTED_SLIDES:
        raise ValueError(f"Expected {EXPECTED_SLIDES} slide sections, found {len(slides_data)}")
    prs = Presentation(str(template))
    delete_all_slides(prs)
    for data in slides_data:
        BUILDERS[data["number"]](prs, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    print(f"Generated: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md"))
    parser.add_argument("--output", type=Path,
                        default=Path("build/seminar_proposal_kopi.pptx"))
    args = parser.parse_args()
    build(args.template, args.source, args.output)


if __name__ == "__main__":
    main()
