"""
Regenerate the model's introductory architecture figure with matplotlib.

This reproduces the decoder-block diagram and model configuration shown at the top
of the notebook as a clean, watermark-free PNG. All layer names and configuration
values are taken from the model's own config in this folder's notebook.

Run:
    python diagram.py

Output:
    olmo3_7b_architecture.png  (written next to this script)
"""
"""Dev harness v2: compute-then-draw sizing. Render all 5, view, iterate."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.lines import Line2D

import os as _os
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "")
GREY = "#d7d7d7"

# ----------------------------- primitives -----------------------------

def box(ax, cx, cy, w, h, text="", fc="white", ec="black", tc="black",
        fs=12, fw="normal", lw=1.6, rounding=0.12, z=3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                 facecolor=fc, edgecolor=ec, lw=lw, zorder=z))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight=fw, zorder=z + 1)

def plus(ax, cx, cy, r=0.28, z=5):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor="black", lw=1.6, zorder=z))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=15, zorder=z + 1)

def arrow(ax, x0, y0, x1, y1, color="black", lw=1.6, z=2, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=z)

def line(ax, xs, ys, color="black", lw=1.6, z=2, ls="-"):
    ax.add_line(Line2D(xs, ys, color=color, lw=lw, zorder=z, linestyle=ls,
                       solid_capstyle="round"))

def leader(ax, x0, y0, x1, y1, color="black", lw=1.6, z=1):
    ax.add_line(Line2D([x0, x1], [y0, y1], color=color, lw=lw, zorder=z,
                       linestyle=(0, (1, 2)), solid_capstyle="round"))

def rich(ax, x, y, lines, fs=13, ha="left", va="center", sep=3):
    rows = []
    for segs in lines:
        areas = [TextArea(t, textprops=dict(color=c, fontweight=w, fontsize=fs))
                 for (t, c, w) in segs]
        rows.append(HPacker(children=areas, align="baseline", pad=0, sep=0))
    packed = VPacker(children=rows, align={"left": "left", "right": "right",
                                           "center": "center"}[ha], pad=0, sep=sep)
    bx = {"left": 0, "center": 0.5, "right": 1}[ha]
    by = {"center": 0.5, "top": 1, "bottom": 0}[va]
    ax.add_artist(AnnotationBbox(packed, (x, y), frameon=False,
                                 box_alignment=(bx, by), zorder=8))

def title(ax, x, y, lines, color, fs=28):
    for i, t in enumerate(lines):
        ax.text(x, y - i * 0.9, t, ha="center", va="center",
                fontsize=fs if i == 0 else fs * 0.62, fontweight="bold", color=color)

def curly_brace(ax, cx, col, top_key, bot_key, label, color, fs=18):
    """Repeat marker (e.g. '32 x') in the left margin, centered on the block,
    tinted to the transformer-block colour."""
    ymid = (col.top_of(top_key) + col.bot_of(bot_key)) / 2
    ax.text(cx - 2.6, ymid, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=9)

def ffn_panel(ax, cx, cy, act, act_color, tint="#e9e9e9", title_txt=None):
    """SwiGLU/GeGLU detail: 3 linear layers + gated activation. 7.0x4.1 panel."""
    w, h = 7.0, 4.1
    box(ax, cx, cy, w, h, fc=tint, ec="black", lw=1.8, rounding=0.2, z=2)
    bw, bh = 2.25, 0.74
    off = 1.42
    y_bot, y_mid, y_top = cy - 1.32, cy - 0.05, cy + 1.28
    box(ax, cx, y_bot, bw, bh, "Linear layer", fc="white", fs=10.5, z=4)
    box(ax, cx - off, y_mid, bw, bh, act, fc="white", fs=9.8, z=4, fw="bold", tc=act_color)
    box(ax, cx + off, y_mid, bw, bh, "Linear layer", fc="white", fs=10.5, z=4)
    box(ax, cx, y_top, bw, bh, "Linear layer", fc="white", fs=10.5, z=4)
    arrow(ax, cx, y_bot + bh / 2, cx - off, y_mid - bh / 2)          # x -> gate
    arrow(ax, cx, y_bot + bh / 2, cx + off, y_mid - bh / 2)          # x -> up
    ymul = (y_mid + y_top) / 2
    ax.text(cx, ymul, "$\\otimes$", ha="center", va="center", fontsize=14, zorder=6)
    line(ax, [cx - off, cx - off], [y_mid + bh / 2, ymul]); line(ax, [cx - off, cx - 0.18], [ymul, ymul])
    line(ax, [cx + off, cx + off], [y_mid + bh / 2, ymul]); line(ax, [cx + off, cx + 0.18], [ymul, ymul])
    arrow(ax, cx, ymul + 0.18, cx, y_top - bh / 2)
    if title_txt:
        ax.text(cx, cy + h / 2 + 0.32, title_txt, ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")

def moe_panel(ax, cx, cy, acc, n_last, fc="white", tint="#ffffff"):
    """MoE detail: router -> experts -> sum."""
    box(ax, cx, cy, 6.6, 5.2, fc=tint, ec=acc, lw=2.0, rounding=0.2, z=2)
    ax.text(cx + 1.6, cy + 2.1, "MoE layer", ha="center", fontsize=13, fontweight="bold", zorder=6)
    plus(ax, cx, cy + 1.35)
    box(ax, cx - 1.95, cy - 0.25, 2.15, 0.66, "Feed forward", fc="#d8d8d8", fs=9.5, z=4)
    box(ax, cx + 1.95, cy - 0.25, 2.15, 0.66, "Feed forward", fc="#d8d8d8", fs=9.5, z=4)
    ax.text(cx, cy - 0.25, ". . .", ha="center", fontsize=15, fontweight="bold", zorder=6)
    box(ax, cx - 1.95 + 0.82, cy - 0.63, 0.42, 0.32, "1", fc="black", tc="white", fs=8.5, z=6)
    box(ax, cx + 1.95 + 0.82, cy - 0.63, 0.5, 0.32, n_last, fc=acc, tc="white", fs=8.5, z=6)
    box(ax, cx, cy - 1.85, 1.6, 0.66, "Router", fc="white", ec=acc, z=4)
    for sx in (cx - 1.9, cx + 1.9):
        arrow(ax, cx, cy - 1.5, sx, cy - 0.6)
        arrow(ax, sx, cy + 0.1, cx, cy + 1.05)
    arrow(ax, cx, cy - 2.55, cx, cy - 2.2)

# ----------------------------- layout engine -----------------------------

class Column:
    def __init__(self, ax, cx, top, gap):
        self.ax, self.cx, self.y, self.gap = ax, cx, top, gap
        self.items, self.pos, self.hgt = [], {}, {}
    def place(self, rows, bw):
        for r in rows:
            k, kind = r["k"], r.get("kind", "box")
            if kind == "plus":
                h = 0.56; cy = self.y - h / 2; plus(self.ax, self.cx, cy)
            elif kind == "attn":
                h = 1.05; cy = self.y - h / 2
                box(self.ax, self.cx, cy, bw, h, r["t"], fc="#3f3f3f", tc="white", fs=10.3)
            else:
                h = 0.82; cy = self.y - h / 2
                fs = r.get("fs", 11.5)
                if max(len(s) for s in r["t"].split("\n")) > 17:
                    fs = min(fs, 10.4)
                box(self.ax, self.cx, cy, bw, h, r["t"], fc="white",
                    ec=r.get("ec", "black"), fs=fs)
            self.pos[k] = cy; self.hgt[k] = h
            self.items.append((k, cy, h))
            self.y = cy - h / 2 - self.gap
        self.bottom = self.y + self.gap
    def connect(self):
        for (k0, y0, h0), (k1, y1, h1) in zip(self.items, self.items[1:]):
            arrow(self.ax, self.cx, y1 + h1 / 2, self.cx, y0 - h0 / 2)
    def top_of(self, k):  return self.pos[k] + self.hgt[k] / 2
    def bot_of(self, k):  return self.pos[k] - self.hgt[k] / 2


def col_height(rows, gap):
    h = 0.0
    for r in rows:
        h += {"plus": 0.56, "attn": 1.05}.get(r.get("kind", "box"), 0.82)
    return h + gap * (len(rows) - 1)


def outer_and_tint(ax, cx, col, block_top_key, block_bot_key, outer_w, tint_color, tint_ec, tint_w=4.2):
    ot = col.top_of("linout") + 0.45
    ob = col.bot_of("tokemb") - 0.45
    box(ax, cx, (ot + ob) / 2, outer_w, ot - ob, fc=GREY, ec="#9a9a9a", lw=1.6, rounding=0.25, z=0)
    tt = col.top_of(block_top_key) + 0.28
    tb = col.bot_of(block_bot_key) - 0.28
    box(ax, cx, (tt + tb) / 2, tint_w, tt - tb, fc=tint_color, ec=tint_ec, lw=1.4, rounding=0.22, z=1)


def residual(ax, cx, col, plus_key, src_key, lx, gap):
    """Branch the skip line from the arrow gap *below* the sub-block's bottom box,
    so it never merges with the block edge."""
    yb = col.bot_of(src_key) - gap * 0.5
    yp = col.pos[plus_key]
    line(ax, [cx, lx], [yb, yb]); line(ax, [lx, lx], [yb, yp])
    arrow(ax, lx, yp, cx - 0.28, yp)


def finish(fig, name):
    fig.savefig(OUT + name + ".png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", name)


def new_ax(w, h, xl=-0.8):
    fig, ax = plt.subplots(figsize=((w - xl) * 0.7, h * 0.7))
    ax.set_xlim(xl, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax

BW = 3.6; BOT = 3.2  # bottom margin (tokenized text zone)

# ----------------------------- builders -----------------------------

def modern(name, title_lines, acc, tint, tint_ec, rows, block_top, block_bot,
           heads_txt, vocab, emb, left_labels, attn_key, brace_txt, res1_from, res2_from,
           side="ffn", ffn=None, moe_last=None, extra=None, sample_txt="Sample input text",
           slide_txt=None):
    gap = 0.44
    ch = col_height(rows, gap)
    top = BOT + ch
    H = top + 3.1
    W = 20.5
    cx = 5.3
    fig, ax = new_ax(W, H)
    col = Column(ax, cx, top, gap)
    col.place(rows, BW)
    col.connect()
    outer_and_tint(ax, cx, col, block_top, block_bot, 6.6, tint, tint_ec)
    title(ax, cx, H - 0.9, title_lines, acc)
    # tokenized text + sample
    tt_y = col.bot_of("tokemb") - 0.95
    box(ax, cx, tt_y, 2.7, 0.82, "Tokenized text", fc="white")
    arrow(ax, cx, tt_y + 0.41, cx, col.bot_of("tokemb"))
    ax.text(cx, tt_y - 0.9, sample_txt, ha="center", fontsize=12, family="monospace")
    arrow(ax, cx, col.top_of("linout"), cx, col.top_of("linout") + 0.85)
    # residuals (skip lines hug the block, in the tint margin)
    lx = cx - 2.0
    residual(ax, cx, col, "plus1", res1_from, lx, gap)
    residual(ax, cx, col, "plus2", res2_from, lx, gap)
    # repeat marker, tinted to the block colour (left of the skip lines)
    curly_brace(ax, cx, col, block_top, block_bot, brace_txt, tint_ec)
    # left labels -> attn
    ay = col.pos[attn_key]
    for i, (lbl, lw_) in enumerate(left_labels):
        ly = ay + 0.75 - i * 0.95
        box(ax, cx - 4.6, ly, lw_, 0.7, lbl, fc="white", fs=10.5)
        arrow(ax, cx - 4.6 + lw_ / 2, ly, cx - BW / 2 + 0.05, ay + (0.25 - i * 0.35))
    # right annotations
    axr = cx + 3.85
    leader(ax, cx + BW / 2, col.pos["linout"], axr - 0.3, col.pos["linout"] + 0.55)
    rich(ax, axr, col.pos["linout"] + 0.55, [[("Vocabulary size of ", "black", "bold")],
         [(vocab, acc, "bold")]])
    leader(ax, cx + BW / 2, col.pos["tokemb"], axr - 0.3, col.pos["tokemb"] - 0.3)
    rich(ax, axr, col.pos["tokemb"] - 0.3, [[("Embedding", "black", "bold")],
         [("dimension of ", "black", "bold"), (emb, acc, "bold")]])
    leader(ax, cx + BW / 2, ay - 0.15, axr - 0.3, ay - 0.5)
    rich(ax, axr, ay - 0.5, heads_txt, fs=11.5)
    if slide_txt:
        rich(ax, axr, ay + 0.95, slide_txt, fs=12)
    # side panels
    if side == "ffn":
        px, py = 15.9, top - 1.7
        ffn_panel(ax, px, py, ffn["act"], acc, title_txt=ffn.get("title"))
        leader(ax, cx + BW / 2, col.pos.get("ff", ay), px - 3.4, py - 0.3)
        rich(ax, px, py - 2.75, ffn["dims"], ha="center", fs=12)
    elif side == "moe":
        px, py = 15.9, BOT + 3.4
        moe_panel(ax, px, py, acc, moe_last, tint="#ffffff")
        leader(ax, cx + BW / 2, col.pos["moe"], px - 3.3, py + 2.2, color=acc)
        rich(ax, px + 0.2, py - 3.2, extra, ha="center", fs=11)
    elif side == "both":
        px, py = 16.3, top - 1.7
        ffn_panel(ax, px, py, ffn["act"], acc, title_txt=ffn.get("title"))
        rich(ax, px, py - 2.75, ffn["dims"], ha="center", fs=12)
        mx, my = 16.4, BOT + 3.1
        moe_panel(ax, mx, my, acc, moe_last, tint="#ffffff")
        leader(ax, cx + BW / 2, col.pos["moe"], mx - 3.4, my + 2.2, color=acc)
        rich(ax, mx + 0.2, my - 3.3, extra, ha="center", fs=11)
    finish(fig, name)


# ---------------- GPT-2 (custom, LayerNorm, no side panel) ----------------


# ----------------------------- figure -----------------------------

def build_olmo():
    rows = [
        {"k": "linout", "t": "Linear output layer"}, {"k": "fnorm", "t": "Final RMSNorm"},
        {"k": "plus2", "kind": "plus"}, {"k": "post2", "t": "Post-RMSNorm 2", "ec": "#ec4899"},
        {"k": "ff", "t": "Feed forward"}, {"k": "plus1", "kind": "plus"},
        {"k": "post1", "t": "Post-RMSNorm 1", "ec": "#ec4899"},
        {"k": "attn", "kind": "attn", "t": "3:1 (local:global) sliding\nwindow multi-head\nattention"},
        {"k": "tokemb", "t": "Token embedding layer"},
    ]
    modern("olmo3_7b_architecture", ["Olmo 3 7B"], "#ec4899", "#f5c6da", "#e48fb4", rows, "plus2", "attn",
           heads_txt=[[("32", "#ec4899", "bold"), (" attention heads", "black", "bold")],
                      [("32", "#ec4899", "bold"), (" key & value heads", "black", "bold")],
                      [("(regular multi-head attention)", "black", "normal")]],
           vocab="100k", emb="4,096",
           left_labels=[("QKNorm", 1.9), ("RoPE + YaRN", 2.4)], attn_key="attn", brace_txt="32 ×",
           res1_from="attn", res2_from="ff",
           slide_txt=[[("Sliding", "black", "bold")], [("window size ", "black", "bold"), ("4096", "#ec4899", "bold")]],
           side="ffn", ffn={"act": "SiLU activation",
                            "dims": [[("Input size: ", "black", "bold"), ("4096", "#ec4899", "bold")],
                                     [("Intermediate projection", "black", "bold")],
                                     [("size = ", "black", "bold"), ("11,008", "#ec4899", "bold")]]})


if __name__ == "__main__":
    build_olmo()
