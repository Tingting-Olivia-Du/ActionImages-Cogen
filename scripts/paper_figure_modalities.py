"""Build the qualitative figure: several episodes x every generated modality, as filmstrips.

    python scripts/paper_figure_modalities.py --out reports/paper_figure/qualitative

Consumes the `frames_view0.npz` that scripts/modality_demo.py writes -- NOT the per-modality
MP4s. Those are h264 at quality=7, which rings on the segmentation palette's hard colour
boundaries and on the action Gaussians, and a reader would be looking at codec artefacts while we
claim to show model output.

LAYOUT. One block per episode; within a block, one row per stream and one column per frame, so
each row reads as a filmstrip and the figure shows that these are videos rather than stills. The
rows are, in order:

  RGB              taken from the `video+action` run. Every panel generates its own RGB -- the
                   four runs are four separate generations that share nothing but the episode --
                   so showing one of them and saying which is the honest option; showing four
                   near-identical RGB rows would waste the space and imply a consistency we did
                   not measure.
  action / depth / segmentation / normal   the auxiliary segment of each run.
  rgb_gt           ground-truth RGB, included only when --no-gt is off.

"GENERATED" IS A COLUMN HEADER, NOT A ROW LABEL. Every row except the optional ground-truth one
is model output, so tagging one row "generated" reads as if the others were not. The word spans
the frame columns once, and the row labels carry only the prompt tag that produced them.

WHY THE RAW CANVAS AND NOT A DECODED VISUALISATION. Depth is shown as the RGB-cube path image the
model actually draws, not as a turbo-mapped metric depth; segmentation as the palette image, not
as recoloured class indices. The claim of the paper is that these modalities are produced in
pixel space by one head, and a decoded rendering would quietly substitute our decoder's output
for the model's. Decoded metrics belong in the tables, not here.

The t=0 column is marked GIVEN, because under IIII it is a condition rather than a prediction and
an unmarked figure would overstate what was generated.
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import img2pdf
except ImportError:                                  # pragma: no cover
    img2pdf = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (npz key, row label). `action` first so the policy stream leads, then the perception streams in
# the canonical order of training/templates.py.
ROWS = [
    ("rgb_gt",                 "RGB (GT)"),
    ("action__rgb_pred",       "RGB"),
    ("action__pred",           "<action>"),
    ("depth__pred",            "<depth>"),
    ("segmentation__pred",     "<scene-seg>"),
    ("normal__pred",           "<normal>"),
]

BG = (255, 255, 255)
INK = (20, 20, 22)
MUTED = (120, 120, 128)
GEN = (14, 110, 60)
GIVEN = (190, 90, 20)


def _save_pdf(canvas, path, dpi):
    """Write a LOSSLESS raster PDF.

    Pillow's own PDF writer encodes RGB as DCT (JPEG) with no way to ask for anything else, and
    JPEG is the wrong codec for this figure: the scene-role palette and the action Gaussians are
    hard-edged saturated colour, exactly what DCT rings on, and a reviewer zooming in would see
    our encoder rather than the model. img2pdf wraps a Flate-compressed PNG instead, so the PDF
    carries the same pixels the PNG does. Pillow is kept as a fallback with a loud warning.
    """
    if img2pdf is None:
        print("  ! img2pdf missing -- falling back to Pillow's LOSSY JPEG-in-PDF encoder "
              "(pip install img2pdf)")
        canvas.convert("RGB").save(path, "PDF", resolution=dpi)
        return
    png_bytes = io.BytesIO()
    canvas.save(png_bytes, format="PNG", optimize=True)
    layout = img2pdf.get_layout_fun(
        (img2pdf.px_to_pt(canvas.width, dpi), img2pdf.px_to_pt(canvas.height, dpi)))
    with open(path, "wb") as f:
        f.write(img2pdf.convert(png_bytes.getvalue(), layout_fun=layout))


def _load(tag_dir):
    z = np.load(os.path.join(tag_dir, "frames_view0.npz"))
    with open(os.path.join(tag_dir, "metrics.json")) as f:
        meta = json.load(f)
    return z, meta


def _block_size(n_rows, n_col, cell, gap, lbl_w, k=1.0):
    S = lambda v: int(round(v * k))
    cap_h, span_h, hdr_h, pad = S(24), S(22), S(20), S(12)
    return (lbl_w + n_col * (cell + gap), cap_h + span_h + hdr_h + n_rows * (cell + gap) + pad)


def _draw_block(draw, canvas, x0, y0, z, meta, have, frames, cell, gap, lbl_w, task, fonts, k=1.0):
    """One episode: caption, the spanning GENERATED header, the t= strip, then the filmstrips."""
    f_task, f_row, f_sm, f_hdr = fonts
    S = lambda v: int(round(v * k))
    n_col = len(frames)
    grid_x = x0 + lbl_w
    grid_w = n_col * (cell + gap) - gap

    y = y0
    instr = meta.get("instruction", "")
    draw.text((x0 + S(6), y + S(2)), task.replace("_", " "), font=f_task, fill=INK)
    tw = draw.textlength(task.replace("_", " "), font=f_task)
    draw.text((x0 + S(6) + tw + S(10), y + S(5)), f'"{instr}"', font=f_sm, fill=MUTED)
    y += S(24)

    # The spanning header. Every row below is model output, so the word belongs here once
    # rather than on one row, where it would imply the other rows are not generated.
    label = "generated"
    lw = draw.textlength(label, font=f_hdr)
    mid = grid_x + grid_w / 2
    draw.line([(grid_x, y + S(11)), (mid - lw / 2 - S(8), y + S(11))], fill=GEN, width=max(1, S(1)))
    draw.line([(mid + lw / 2 + S(8), y + S(11)), (grid_x + grid_w, y + S(11))], fill=GEN, width=max(1, S(1)))
    draw.text((mid - lw / 2, y + S(4)), label, font=f_hdr, fill=GEN)
    y += S(22)

    for c, fi in enumerate(frames):
        x = grid_x + c * (cell + gap)
        if fi == 0:
            draw.text((x + S(2), y + S(4)), "t=0", font=f_sm, fill=GIVEN)
            draw.text((x + S(2) + draw.textlength("t=0", font=f_sm) + S(5), y + S(4)),
                      "given", font=f_sm, fill=GIVEN)
        else:
            draw.text((x + S(2), y + S(4)), f"t={fi}", font=f_sm, fill=MUTED)
    y += S(20)

    for key, lab in have:          # not `k`: that is the scale factor in this scope
        arr = z[key]
        is_gt = key == "rgb_gt"
        draw.text((x0 + S(6), y + cell // 2 - S(8)), lab, font=f_row, fill=MUTED if is_gt else INK)
        for c, fi in enumerate(frames):
            fr = arr[min(fi, len(arr) - 1)]
            im = Image.fromarray(fr).resize((cell, cell), Image.LANCZOS)
            x = grid_x + c * (cell + gap)
            canvas.paste(im, (x, y))
            # the given column is a condition, not a prediction: ring it so the distinction
            # survives the figure being cropped into a slide.
            if fi == 0 and not is_gt:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=GIVEN, width=max(2, S(2)))
        y += cell + gap
    return y + S(12)


def _fonts(k=1.0):
    px = lambda v: max(6, int(round(v * k)))
    return (ImageFont.truetype(FONT_B, px(15)), ImageFont.truetype(FONT, px(13)),
            ImageFont.truetype(FONT, px(11)), ImageFont.truetype(FONT_B, px(12)))


def build(tag_dirs, frames, cell, out_stem, gap=3, rows=None, per_scene=False, scale=1.0,
          base_dpi=200.0):
    """`scale` multiplies pixel density only. Every length and font size is multiplied by it and
    the PDF dpi is multiplied to match, so the figure keeps its physical size on the page and
    gains resolution. At scale 1 a 178 px cell was resampled down from a 512 px source frame and
    threw away 88% of the pixels the model produced; the frames are the whole point of the figure,
    so the default is now to render them at their native size."""
    rows = rows or ROWS
    k = float(scale)
    cell = int(round(cell * k))
    gap = max(1, int(round(gap * k)))
    lbl_w = int(round(118 * k))
    dpi = base_dpi * k
    blocks = []
    for d in tag_dirs:
        z, meta = _load(d)
        have = [(k, lab) for k, lab in rows if k in z.files]
        missing = [k for k, _ in rows if k not in z.files]
        if missing:
            print(f"  ! {os.path.basename(d)}: missing {missing}")
        blocks.append((d, z, meta, have))

    fonts = _fonts(k)
    written = []
    if per_scene:
        # One file per scene. A combined sheet is unusable as a LaTeX float once it is taller
        # than a page, and a per-scene file lets the paper place them independently.
        for d, z, meta, have in blocks:
            task = os.path.basename(d).replace("fig_", "")
            w, h = _block_size(len(have), len(frames), cell, gap, lbl_w, k)
            m = int(round(4 * k))
            canvas = Image.new("RGB", (w + 2 * m, h + m), BG)
            draw = ImageDraw.Draw(canvas)
            _draw_block(draw, canvas, m, m, z, meta, have, frames, cell, gap, lbl_w, task, fonts, k)
            png, pdf = f"{out_stem}_{task}.png", f"{out_stem}_{task}.pdf"
            canvas.save(png)
            _save_pdf(canvas, pdf, dpi)
            written += [png, pdf]
            print(f"  wrote {os.path.basename(png)} / {os.path.basename(pdf)}  "
                  f"({canvas.width}x{canvas.height} px, cell {cell}px, {dpi:.0f} dpi, "
                  f"{canvas.width / dpi:.2f}x{canvas.height / dpi:.2f} in)")
        return written

    sizes = [_block_size(len(h), len(frames), cell, gap, lbl_w, k) for _, _, _, h in blocks]
    m = int(round(4 * k))
    W = max(w for w, _ in sizes) + 2 * m
    H = sum(h for _, h in sizes) + 2 * m
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    y = m
    for (d, z, meta, have), (_, bh) in zip(blocks, sizes):
        task = os.path.basename(d).replace("fig_", "")
        y = _draw_block(draw, canvas, m, y, z, meta, have, frames, cell, gap, lbl_w, task, fonts, k)
        draw.line([(m, y - int(round(6 * k))), (W - 2 * m, y - int(round(6 * k)))],
                  fill=(226, 226, 231), width=max(1, int(round(k))))
    canvas.save(out_stem + ".png")
    _save_pdf(canvas, out_stem + ".pdf", dpi)
    print(f"  wrote {out_stem}.png / .pdf  ({canvas.width}x{canvas.height} px, cell {cell}px, "
          f"{dpi:.0f} dpi, {canvas.width / dpi:.2f}x{canvas.height / dpi:.2f} in)")
    return [out_stem + ".png", out_stem + ".pdf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default=os.path.join(REPO, "reports", "paper_figure"),
                    help="parent holding fig_<task>/ directories, or a comma list of them")
    ap.add_argument("--tags", default="", help="comma list of tag names to include, in order")
    ap.add_argument("--frames", default="0,8,16,24,32,40")
    ap.add_argument("--no-gt", action="store_true",
                    help="drop the ground-truth RGB row. The figure then shows only what the "
                         "model produced, which is the compact form; the t=0 GIVEN marking still "
                         "records that the first column is a condition rather than a prediction.")
    ap.add_argument("--cell", type=int, default=132)
    ap.add_argument("--no-rgb", action="store_true",
                    help="also drop the generated-RGB row, leaving only the four auxiliary streams")
    ap.add_argument("--per-scene", action="store_true",
                    help="one PNG+PDF per episode instead of one combined sheet")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="pixel-density multiplier. Lengths, fonts and the PDF dpi are all scaled "
                         "together, so the figure keeps its size on the page and only gains "
                         "resolution. Source frames are 512px, so scale = 512/cell renders them "
                         "natively and anything beyond that only upsamples.")
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "paper_figure", "qualitative"))
    a = ap.parse_args()

    if "," in a.dirs:
        dirs = [d.strip() for d in a.dirs.split(",")]
    elif a.tags:
        dirs = [os.path.join(a.dirs, t.strip()) for t in a.tags.split(",")]
    else:
        dirs = sorted(glob.glob(os.path.join(a.dirs, "fig_*")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "frames_view0.npz"))]
    if not dirs:
        raise SystemExit(f"no directories with frames_view0.npz under {a.dirs}")
    print(f"episodes: {[os.path.basename(d) for d in dirs]}")
    frames = [int(x) for x in a.frames.split(",")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    drop = set()
    if a.no_gt:
        drop.add("rgb_gt")
    if a.no_rgb:
        drop.add("action__rgb_pred")
    rows = [r for r in ROWS if r[0] not in drop]
    build(dirs, frames, a.cell, a.out, rows=rows, per_scene=a.per_scene, scale=a.scale)


if __name__ == "__main__":
    main()
