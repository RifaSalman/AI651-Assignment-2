"""Build the Assignment 2 report PDF."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Question2" / "kaggle_output" / "outputs"
SCORES = json.loads((FIG / "baseline_scores.json").read_text(encoding="utf-8"))
PHASE2_PATH = FIG / "phase2_scores.json"
OUT_PDF = ROOT / "report" / "AI651_Assignment2_RifaSalman.pdf"
REPO = "https://github.com/RifaSalman/AI651-Assignment-2"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
FONT_B = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
FONT_I = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Oblique.ttf"


def fmt(value: float) -> str:
    if value != value:
        return "—"
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-3:
        text = f"{value:.2e}"
        return text.replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.3g}"


def pair(a: float, b: float) -> str:
    return f"{fmt((a + b) / 2)} ({fmt(a)}, {fmt(b)})"


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=16)
        self.add_font("DejaVu", "", str(FONT))
        self.add_font("DejaVu", "B", str(FONT_B))
        self.add_font("DejaVu", "I", str(FONT_I))
        self.set_margin(16)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 6, str(self.page_no()), align="C")
        self.set_text_color(0, 0, 0)

    def _write(self, text: str, height: float):
        self.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")

    def h1(self, text: str):
        self.ln(2)
        self.set_font("DejaVu", "B", 14)
        self._write(text, 7)
        self.ln(1)

    def h2(self, text: str):
        self.ln(1)
        self.set_font("DejaVu", "B", 12)
        self._write(text, 6)
        self.ln(1)

    def h3(self, text: str):
        self.ln(1)
        self.set_font("DejaVu", "B", 11)
        self._write(text, 6)

    def body(self, text: str):
        self.set_font("DejaVu", "", 10)
        self._write(text, 5)
        self.ln(1.5)

    def bullet(self, text: str):
        self.set_font("DejaVu", "", 10)
        self._write("•  " + text, 5)
        self.ln(0.4)

    def caption(self, text: str):
        self.set_font("DejaVu", "I", 9)
        self._write(text, 4.4)
        self.ln(2)

    def ensure(self, height: float):
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def figure(self, path: Path, caption: str, width: float = 178):
        self.ensure(70)
        self.image(str(path), w=width)
        self.ln(1)
        self.caption(caption)

    def add_table(self, headers: list[str], rows: list[list[str]], font: float = 7.5):
        self.set_font("DejaVu", "", font)
        with super().table(
            width=self.epw,
            line_height=4.2,
            text_align="CENTER",
            first_row_as_headings=True,
        ) as table:
            head = table.row()
            for name in headers:
                head.cell(name)
            for data in rows:
                line = table.row()
                for item in data:
                    line.cell(item)
        self.ln(2)


def render_q1(pdf: Report):
    text = (ROOT / "report" / "q1.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            pdf.h1(line[2:].strip())
        elif line.startswith("## "):
            pdf.h2(line[3:].strip())
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            cells = []
            for raw in block:
                if set(raw.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    continue
                cells.append([part.strip() for part in raw.strip("|").split("|")])
            if cells:
                pdf.add_table(cells[0], cells[1:])
            continue
        elif line.startswith("- "):
            pdf.bullet(line[2:].strip())
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ")):
                para.append(lines[i].rstrip())
                i += 1
            pdf.body(" ".join(para))
            continue
        i += 1
    pdf.figure(
        ROOT / "report" / "figures" / "q1_architecture.png",
        "Figure Q1. Forecasting architecture. Blue boxes are observations, sand boxes are fixed by the river, green boxes are learned, and the issued product is the +60 min level at S1–S4.",
    )


def learned_row(block: dict, key: str) -> str:
    return pair(block[0][key], block[1][key])


def q2(pdf: Report):
    a = SCORES["A"]
    blank = SCORES["reference"]["Blank"]
    copy = SCORES["reference"]["Copy-last"]
    pdf.add_page()
    pdf.h1("Question 2 — Motion or Memory?")
    pdf.body(
        "Hardware for every learned model is an NVIDIA Tesla T4 on Kaggle. "
        "Training time is the mean wall-clock time per epoch over 30 epochs, excluding the dataset download. "
        "Inference time is the wall-clock time to score all 1,000 original test videos at batch size 32, "
        "after each batch is on the device, under model.eval() and torch.no_grad(). "
        "Seeds 0 and 1 are reported as mean (seed 0, seed 1). Figures and GIFs use seed 0. "
        "Model A took 11.9 s (11.9 s, 11.9 s) per epoch."
    )

    pdf.h2("Table 1. Transformation and window checks")
    pdf.add_table(
        ["Check", "Expected", "Result"],
        [
            ["(a) max |M(M(x)) − x|", "0", "0, torch.equal True"],
            ["(b) max |R^4(x) − x|", "0", "0, torch.equal True"],
            ["(c) max |M(R(M(x))) − R^{−1}(x)|", "0", "0, torch.equal True"],
            ["(d) bright pixel of M(p)", "(7, 10, 43)", "(7, 10, 43)"],
            ["(e) bright pixel of R(p)", "(7, 43, 10)", "(7, 43, 10)"],
            ["(f) 1× input / target", "1–5 / 6–10", "1–5 / 6–10"],
            ["(g) 2× input / target", "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", "same"],
            ["(h) 3× input / target", "1, 4, 7, 10, 13 / 16, 19", "same"],
        ],
        font=8,
    )
    pdf.caption(
        "Table 1. Checks (a)–(h) on the functions used in the experiments. "
        "x is torch.rand(20, 1, 64, 64) after torch.manual_seed(0). "
        "p is zero except p[7, 0, 10, 20] = 1. Model A has 480,784 parameters."
    )

    pdf.figure(
        FIG / "figure1.png",
        "Figure 1. Optical flow on the first 50 training videos, 950 frame pairs, Farneback settings as specified. "
        "Pixels kept: intensity above 0.2 on both frames and magnitude above 0.5. "
        f"Kept pixels: {json.loads((FIG / 'figure1_stats.json').read_text())['n_pixels']}. "
        "Median speed 3.32 pixels per frame. The horizontal line in (a) is 1/18.",
    )
    pdf.h3("Q2.1")
    pdf.body(
        "Mirrored and rotated trajectories are partly familiar, and they are not exchangeable with the training distribution. "
        "Every 20° bin in Figure 1(a) is occupied, so a reflected or rotated direction is not a new kind of motion. "
        "The bins are not flat: the fullest holds 0.082 of the kept pixels and the emptiest 0.026, a ratio of 3.11, against a uniform level of 1/18 ≈ 0.0556. "
        "Reflecting every direction moves about 0.13 of the probability mass (total variation between the histogram and its mirror). "
        "A 90° rotation is not aligned to the 20° bins; the closest shifts, 4 and 5 bins, move about 0.18 of the mass. "
        "A typical speed is the median, 3.32 pixels per frame (mean 3.27, 90th percentile 4.12). "
        "Twice that median is 6.63 pixels per frame. Three times it is 9.95. Twice the 90th percentile is 8.25. "
        "Optical flow is unreliable where digits overlap. The kept set already drops pixels that are dark in either frame, but overlap boundaries can still invent vectors and fatten the tails. "
        "That bias would have to move the median a long way to change the comparison with an 8-pixel reach. It does not: even a median of 4 would put 2× on the reach and 3× beyond it."
    )
    pdf.h3("Q2.2")
    pdf.body(
        "A mirror changes both the appearance and the motion statistics. A mirrored digit is a different shape, and the direction histogram is reflected. "
        "A 90° rotation changes appearance more than a mirror does, and it sends every direction into a different part of Figure 1(a). "
        "A speed-up does not change the digit. It only increases the displacement between the frames the model sees. "
        "The three therefore separate appearance, an unfamiliar direction, and the size of the step. "
        "Constant-velocity motion with elastic bounces is still valid motion after all three, so a model that had learned the motion rule should cope with all of them."
    )

    pdf.h2("Table 2. Baseline forecast error")
    headers = ["Model", "MSE1", "MSE2", "MSE3", "MSE4", "MSE5", "E(5)", "Parameters"]
    orig = a["original"]
    rows = [[
        "A",
        learned_row(orig, "MSE1"),
        learned_row(orig, "MSE2"),
        learned_row(orig, "MSE3"),
        learned_row(orig, "MSE4"),
        learned_row(orig, "MSE5"),
        learned_row(orig, "E5"),
        "480,784",
    ]]
    for name, block in (("Blank", blank), ("Copy-last", copy)):
        o = block["original"]
        rows.append([name, fmt(o["MSE1"]), fmt(o["MSE2"]), fmt(o["MSE3"]), fmt(o["MSE4"]), fmt(o["MSE5"]), fmt(o["E5"]), "0"])
    pdf.add_table(headers, rows, font=7)
    pdf.caption("Table 2. Test-set MSE by horizon at 1×, t = 1. Learned entries are mean (seed 0, seed 1).")

    pdf.figure(
        FIG / "figure2.png",
        "Figure 2. Seed-0 model A on test videos 9000, 9007 and 9012. "
        "Each block is input, prediction, ground truth, five frames each. "
        "Video 9000 is the overlapping pair (a 5 and a 6 merge and separate). "
        "Videos 9007 and 9012 keep two components through frames 1–10.",
    )
    pdf.h3("Q2.3")
    pdf.body(
        "pixel_unshuffle(4) makes each cell of the 16×16 grid stand for 4 original pixels, centre to centre. The 4×4 patch width is not added. "
        "The 1×1 head, the unshuffle and the shuffle do not move information. "
        "Each ConvLSTM layer is a 3×3 convolution with dilation 1, so it reads offsets −1, 0 and +1 and adds a Chebyshev radius of one cell. "
        "In one step the new input passes through layer 1 and then layer 2. Layer 1’s previous hidden state takes that same path. "
        "Layer 2’s previous state passes through layer 2 only, which is shorter. "
        "The longest path is 2 cells, 8 original pixels along a row or a column. "
        "The corner of that neighbourhood is 8√2 ≈ 11.3 pixels. The reach reported here is the axis-aligned one, 8 pixels."
    )
    pdf.h3("Q2.4")
    pdf.body(
        "Model A’s E(5) is 0.0223, against 0.0426 for blank and 0.0641 for copy-last. "
        "It is clearly better than both references on ordinary 1× clips. "
        "The error grows with the horizon, from MSE1 = 0.0171 to MSE5 = 0.0263. "
        "That is the closed-loop rollout: each later frame is conditioned on the network’s own previous prediction, so blur accumulates. "
        "In Figure 2 the separated digits in videos 9007 and 9012 stay recognisable and follow the right general direction, while becoming softer. "
        "Video 9000 is the overlap. The 5 and the 6 merge, and the forecast turns them into one soft blob instead of recovering two digits the way the ground truth does. "
        "Overlap is where the model loses which digit is which, not only where it is."
    )
    pdf.h3("Q2.5")
    pdf.body((ROOT / "Question2" / "predictions" / "Q2.5.txt").read_text(encoding="utf-8").strip())

    pdf.h2("Table 3. Mirror and rotation")
    pdf.add_table(
        ["Model", "E(5)", "C_M", "E_M", "C_R", "E_R"],
        [
            ["A", learned_row(orig, "E5"), pair(a["CM"][0], a["CM"][1]), pair(a["EM"][0]["E5"], a["EM"][1]["E5"]), pair(a["CR"][0], a["CR"][1]), pair(a["ER"][0]["E5"], a["ER"][1]["E5"])],
            ["Blank", fmt(blank["original"]["E5"]), "0", fmt(blank["mirror_error"]["E5"]), "0", fmt(blank["rotation_error"]["E5"])],
            ["Copy-last", fmt(copy["original"]["E5"]), "0", fmt(copy["mirror_error"]["E5"]), "0", fmt(copy["rotation_error"]["E5"])],
        ],
        font=8,
    )
    pdf.caption("Table 3. Consistency gaps and transformed error at 1×, t = 1. Blank and copy-last are exactly consistent. Their C_M and C_R are 0.")

    pdf.figure(
        FIG / "figure3.png",
        "Figure 3. Seed-0 model A, the two test videos with the largest per-video C_M (9642, 9279) and the two with the largest per-video C_R (9168, 9925). "
        "Rows in each panel are X | f(X); T(X) | T(Y); T(X) | f(T(X)); T(X) | T(f(X)). "
        "GIFs, one horizon at a time: Question2/outputs/gifs/figure3_mirror_0.gif, figure3_mirror_1.gif, figure3_rotation_0.gif, figure3_rotation_1.gif.",
    )
    pdf.figure(
        FIG / "figure4.png",
        "Figure 4. Per-video C_M and C_R of seed-0 model A on the 1,000 test videos. Red lines mark the videos in Figure 3.",
    )
    pdf.h3("Q2.6")
    pdf.body(
        "C_T = 0 means the forecast commutes with T. It does not mean the forecast is right. "
        "Blank and copy-last commute for a structural reason: a black frame is still black after a mirror or a rotation, and repeating the last frame commutes with a pixel permutation. "
        "Their transformed error equals their original error, 0.0426 and 0.0641, so they are consistent and wrong. "
        "E_T has to sit next to C_T. Otherwise a constant predictor looks perfect."
    )
    pdf.h3("Q2.7")
    pdf.body(
        "Video 9168, the largest rotation gap for seed 0 (C_R = 0.00572). "
        "Rows 3 and 4 of that panel share the rotated input. "
        "Row 4 is the original forecast after the same rotation: both digits are still there, blurred, and they keep moving apart. "
        "Row 3 is a fresh forecast from the rotated input: one digit fades within the five horizons and the other smears into a block. "
        "The model’s account of the motion changes. Rotating a forecast it was willing to make on the upright clip does not match the forecast it makes once the clip is rotated. Shown the rotated clip, it loses a digit instead of continuing both trajectories."
    )
    pdf.h3("Q2.8")
    pdf.body(
        "Yes. Mean C_R is 0.00293 against mean C_M of 0.00174, and E_R is 0.0239 against E_M of 0.0229 and an original E(5) of 0.0223. "
        "Figure 4 sits further to the right for rotation than for mirror. "
        "From Q2.1, a 90° rotation moves more of the direction histogram than a mirror does. From Q2.2, it also changes the shape of a digit more. "
        "Both of those are about appearance and about which directions were common, not about a different law of motion. "
        "The transformed errors stay far below the blank predictor (0.0426). The model is still forecasting. "
        "The speed test, which changes the step and not the appearance, is the one that reaches the blank error. That is a different failure."
    )

    pdf.h2("Table 4. Faster motion")
    def speed_pair(block, k, key):
        return pair(block["speed"][k][0][key], block["speed"][k][1][key])

    pdf.add_table(
        ["Model", "E(2) at 1×", "E(2) at 2×", "E(2) at 3×", "E(5) at 1×", "E(5) at 2×"],
        [
            ["A", speed_pair(a, "1", "E2"), speed_pair(a, "2", "E2"), speed_pair(a, "3", "E2"), speed_pair(a, "1", "E5"), speed_pair(a, "2", "E5")],
            ["Copy-last", fmt(copy["speed"]["1"]["E2"]), fmt(copy["speed"]["2"]["E2"]), fmt(copy["speed"]["3"]["E2"]), fmt(copy["speed"]["1"]["E5"]), fmt(copy["speed"]["2"]["E5"])],
        ],
        font=8,
    )
    pdf.caption("Table 4. Speed comparison on the test set at t = 1. E(5) is not defined at 3×, because only two target frames exist. Blank’s E(2) is 0.0426, 0.0425 and 0.0426 at 1×, 2× and 3×.")
    pdf.figure(
        FIG / "figure5.png",
        "Figure 5. Seed-0 model A on test video 9021, chosen so the displayed true frames do not overlap. "
        "Each row is one speed: five input frames, the first two predictions, and the first two true future frames, at 1×, 2× and 3×.",
    )
    pdf.h3("Q2.9")
    pdf.body(
        "Model A falls apart at 2×, and 3× is worse. E(2) goes from 0.0188 at 1× to 0.0449 at 2× and 0.0523 at 3×. "
        "Blank’s E(2) is about 0.0426, so at 2× the forecast is already no better than predicting black. "
        "This is earlier than the median argument in Q2.5. Twice the median speed is 6.63 pixels, inside a reach of 8, and three times the median is 9.95, outside it. "
        "What that estimate missed is the tail and the width of a digit. Twice the 90th percentile is 8.25 pixels, already past the axis-aligned reach, so a real share of 2× motion cannot be joined to the previous position in one update. A digit is also wider than the one pixel whose flow was measured. "
        "Figure 5, video 9021, shows the failure as smear. At 1× the 1 and the 4 stay sharp enough to follow. At 2× and 3× the forecast is a soft ghost near the last input position, while the true digits have moved on. "
        "A stall would repeat a sharp digit. A duplicate would show two copies. A vanishing digit would go to black. This is the smear: the model no longer has a position to commit to."
    )
    pdf.h3("Q2.10")
    pdf.body(
        "No. Model A never trained on k = 2 or k = 3, and 2× motion also sits on the edge of the 8-pixel reach. Either fact can produce Figure 5. "
        "The split is model C, the same architecture trained with k drawn from {1, 2, 3}, against model D, a larger reach trained on the same speeds. "
        "If C repairs 2× and still fails at 3×, then 2× was missing from the training distribution and 3× is beyond the architecture. "
        "If C fails at 2× as well, the reach was already the limit, and more fast clips cannot move information further than the convolutions allow."
    )
    pdf.h3("Q2.11")
    pdf.body((ROOT / "Question2" / "predictions" / "Q2.11.txt").read_text(encoding="utf-8").strip())
    pdf.h3("Q2.12")
    pdf.body(
        "Model D is model A’s ConvLSTM with one change. Layer 1 stays a 3×3 with dilation 1. Layer 2 is a 3×3 with dilation 3 and padding 3, so it reads offsets −3, 0 and +3. "
        "Dilation adds no weights, so the parameter count stays 480,784, under the limit of 721,176. There is no attention and no extra layer. "
        "The 9-step rollout, the shapes, the loss and the rest of the training protocol are unchanged. The only protocol change is the one in the model table: k is drawn from {1, 2, 3}. "
        "Reach uses the same rules as Q2.3. One cell is 4 original pixels. Layer 1 adds a radius of 1 cell and layer 2 adds a radius of 3. "
        "The longest path, the new input or layer 1’s previous state through both convolutions, has Chebyshev radius 4 cells, which is 16 original pixels. "
        "The summed offsets are every integer from −4 to 4, so that neighbourhood has no holes. "
        "Layer 2’s own previous state only reaches 12 pixels, which is the shorter path. "
        "16 is strictly larger than model A’s 8. Three times the median speed, 9.95, and three times the 90th percentile, about 12.4, are inside 16 and outside 8."
    )


def main():
    gif_dir = ROOT / "Question2" / "outputs" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "figure3_mirror_0.gif",
        "figure3_mirror_1.gif",
        "figure3_rotation_0.gif",
        "figure3_rotation_1.gif",
        "figure6_DB_mirror_0.gif",
        "figure6_DB_mirror_1.gif",
        "figure6_DB_rotation_0.gif",
        "figure6_DB_rotation_1.gif",
    ):
        source = FIG / "gifs" / name
        if source.exists():
            shutil.copy(source, gif_dir / name)
    for name in (
        "figure1.png",
        "figure2.png",
        "figure3.png",
        "figure4.png",
        "figure5.png",
        "figure6a_DB.png",
        "figure6b_DB.png",
    ):
        source = FIG / name
        if source.exists():
            shutil.copy(source, ROOT / "Question2" / "outputs" / name)

    pdf = Report()
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf._write("AI651 — Deep Learning for Space, Time and Graphs", 8)
    pdf.set_font("DejaVu", "", 12)
    pdf._write("Assignment 2", 6)
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 11)
    pdf.body("Rifa Salman")
    pdf.body("Fall 2026")
    pdf.body("Code: " + REPO)
    pdf.body(
        "Question 1 is a forecasting design and was not trained. "
        "Question 2 was trained on an NVIDIA Tesla T4. "
        "Learned numbers are mean (seed 0, seed 1). Errors are three significant figures. Values below 10⁻³ are in scientific notation."
    )
    render_q1(pdf)
    q2(pdf)
    if PHASE2_PATH.exists():
        add_phase2(pdf, json.loads(PHASE2_PATH.read_text(encoding="utf-8")))
    pdf.add_page()
    pdf.h1("AI use")
    pdf.body(
        "This assignment was completed with Cursor’s coding agent (Grok 4.7). "
        "The prompts were the assignment PDF, the instruction to finish both questions without further questions, "
        "the instruction to run Question 2 on Kaggle, and a Kaggle API token used only to authenticate the Kaggle CLI. "
        "The agent wrote the ConvLSTM code, the checks, the training and evaluation scripts, the Question 1 design, the figures, and this report. "
        "Training ran on Kaggle. The paragraphs under Q2.5 and Q2.11 are the files committed before those experiments and were not edited afterwards. "
        "Numbers in the tables were taken from the saved score files. "
        "The architecture diagram was drawn for this design. The Moving MNIST figures were produced by the evaluation code."
    )
    pdf.output(str(OUT_PDF))
    print(OUT_PDF)


def _cell(rows: list[dict], *keys: str) -> str:
    values = []
    for row in rows:
        item = row
        for key in keys:
            item = item[key]
        values.append(float(item))
    return f"{fmt(sum(values) / 2)}\n({fmt(values[0])}, {fmt(values[1])})"


def _one(rows: list[dict], *keys: str) -> str:
    return _cell(rows, *keys).replace("\n", " ")


def add_phase2(pdf: Report, payload: dict):
    models = payload["models"]
    wrapped = payload["wrapped"]
    final = payload["final"]
    pdf.add_page(orientation="L")
    pdf.h2("Table 5. What changes with the data, and with the reach")
    headers = ["Model", "E(5)", "C_M", "E_M", "C_R", "E_R", "E(2) 1×", "E(2) 2×", "E(2) 3×", "Params", "Train s", "Infer s"]
    rows = []
    for name in ("A", "B", "C", "D", "DB"):
        block = models[name]
        rows.append([
            "D+B" if name == "DB" else name,
            _cell(block, "original", "E5"),
            _cell(block, "CM"),
            _cell(block, "EM", "E5"),
            _cell(block, "CR"),
            _cell(block, "ER", "E5"),
            _cell(block, "speed", "1", "E2"),
            _cell(block, "speed", "2", "E2"),
            _cell(block, "speed", "3", "E2"),
            f"{block[0]['parameters']:,}",
            _cell(block, "epoch_seconds_mean"),
            _cell(block, "inference_seconds"),
        ])
    pdf.add_table(headers, rows, font=7)
    pdf.caption(
        "Table 5. Test metrics for the learned models. D+B is model D trained with both speed and symmetry augmentation. "
        "Train s is the mean seconds per epoch. Infer s is one forward pass of the 1,000 test videos after the batch is on the Tesla T4. "
        "Each learned entry is mean, then seed 0 and seed 1."
    )

    pdf.h3("Q2.13")
    pdf.body(
        "The mirror and rotation gaps are an appearance failure, in the sense of Q2.2: the motion law is unchanged and the digits are not. "
        f"Model B, which only adds g drawn from G, cuts C_M from {_one(models['A'], 'CM')} to {_one(models['B'], 'CM')} "
        f"and C_R from {_one(models['A'], 'CR')} to {_one(models['B'], 'CR')}. "
        f"Its transformed errors fall into line with its own E(5) of {_one(models['B'], 'original', 'E5')}. "
        f"Model C, which changes speed and not symmetry, leaves C_R at {_one(models['C'], 'CR')}, close to model A. "
        "So the symmetry gap moves when the training distribution is symmetrised, and it does not move when only the speeds change. "
        "The speed failure is unfamiliar motion, not architectural reach. "
        f"Model C has the same 8-pixel reach as A and cuts E(2) at 2× from {_one(models['A'], 'speed', '2', 'E2')} to {_one(models['C'], 'speed', '2', 'E2')} "
        f"and at 3× from {_one(models['A'], 'speed', '3', 'E2')} to {_one(models['C'], 'speed', '3', 'E2')}. "
        f"Both of those are better than blank, which sits near 0.0426. "
        f"Model D, with reach 16 and the same speed augmentation, does not beat C: its 3× E(2) is {_one(models['D'], 'speed', '3', 'E2')}. "
        "The conditional in the question does not hold. C is not failing at 3× while D succeeds. "
        "More examples of fast motion were enough, inside the original reach. "
        "That does not mean data can extend a reach. A 3×3 kernel has a zero weight on every pixel outside its offsets, for every value of the weights. "
        "Training changes those weights. It does not change which pixels are read. "
        "D was the test of that limit, and on this data the limit was not what separated the errors."
    )
    pdf.h3("Q2.14")
    pdf.body(
        f"Yes. Original E(5) goes from {_one(models['A'], 'original', 'E5')} for A to {_one(models['B'], 'original', 'E5')} for B, "
        f"{_one(models['C'], 'original', 'E5')} for C, {_one(models['D'], 'original', 'E5')} for D and {_one(models['DB'], 'original', 'E5')} for D+B. "
        "B’s cost is small beside the drop in the consistency gaps, and it is worth it. "
        "C’s cost is larger, and it is the cost of no longer collapsing to the blank predictor at 2× and 3×. "
        "That trade is worth it for the question being asked. "
        "D’s further cost over C bought no speed improvement. The dilation was not worth it on these clips. "
        "D+B matches D on the original clips and is the better of those two under mirror and rotation, which is why it is the one wrapped below."
    )

    pdf.h2("Table 6. Symmetrised forecasters")
    pdf.body(
        "For any g in G and any T in {M, R}, right-multiplication by T is a bijection of G. "
        "Set h = g T, so g^{−1} = T h^{−1}. Then "
        "f̄(TX) = (1/8) Σ_g g^{−1} f(g T X) = (1/8) Σ_h T(h^{−1} f(h X)) = T(f̄(X)), "
        "because T permutes pixels and therefore passes through the average. "
        "So f̄(TX) = T(f̄(X)), and C_M = C_R = 0 for every underlying f. "
        "The values below are the floating-point remainder."
    )
    wrows = []
    labels = {"B": "sym B", "D": "sym D", "DB": "sym D+B (final)"}
    for name in ("B", "D", "DB"):
        block = wrapped[name]
        wrows.append([
            labels[name],
            _cell(block, "original", "E5"),
            _cell(block, "CM"),
            _cell(block, "EM", "E5"),
            _cell(block, "CR"),
            _cell(block, "ER", "E5"),
            _cell(block, "speed", "1", "E2"),
            _cell(block, "speed", "2", "E2"),
            _cell(block, "speed", "3", "E2"),
            f"{block[0]['parameters']:,}",
            _cell(block, "epoch_seconds_mean"),
            _cell(block, "inference_seconds"),
        ])
    pdf.add_table(headers, rows=wrows, font=7)
    pdf.caption(
        "Table 6. The same columns as Table 5, for the symmetrised forecasters. "
        "Train s is the underlying learned model. Symmetrisation is not trained. "
        "C_M and C_R are floating-point zeros and are written in scientific notation. "
        f"The final model is symmetrised D+B, not symmetrised D. "
        f"D+B’s original E(5) stays within 5% of D’s, and both of its transformed errors are lower, so Table 5 supports wrapping D+B."
    )

    pdf.add_page(orientation="P")
    pdf.figure(
        FIG / "figure6a_DB.png",
        "Figure 6(a). The final model, symmetrised D+B, seed 0, on the same videos as Figure 3: mirror 9642 and 9279, rotation 9168 and 9925. "
        "Rows match Figure 3. GIFs: Question2/outputs/gifs/figure6_DB_mirror_0.gif, figure6_DB_mirror_1.gif, figure6_DB_rotation_0.gif, figure6_DB_rotation_1.gif.",
    )
    pdf.figure(
        FIG / "figure6b_DB.png",
        "Figure 6(b). The same final model on test video 9021, the video in Figure 5, at 1×, 2× and 3×.",
    )
    pdf.h3("Q2.15")
    pdf.body(
        f"Learned symmetry is model B. Its C_M is {_one(models['B'], 'CM')} and its C_R is {_one(models['B'], 'CR')}, "
        f"against {_one(models['A'], 'CM')} and {_one(models['A'], 'CR')} for A. "
        f"Its E_M and E_R match its E(5) of {_one(models['B'], 'original', 'E5')}. The gap shrinks. It does not become zero. "
        f"Inference stays one forward pass, {_one(models['B'], 'inference_seconds')} s. "
        "Built-in symmetry is the average over G. It is the exact one, because the short argument above does not depend on f. "
        f"Wrapped B has C_M = {_one(wrapped['B'], 'CM')} and C_R = {_one(wrapped['B'], 'CR')}. "
        f"The cost is the eight passes: {_one(wrapped['B'], 'inference_seconds')} s against {_one(models['B'], 'inference_seconds')} s, and "
        f"{_one(wrapped['DB'], 'inference_seconds')} s for the final model against {_one(models['DB'], 'inference_seconds')} s for D+B unwrapped. "
        "Averaging eight forecasts blurs them when they disagree about where a digit is. "
        f"Here they mostly agree. Wrapped B’s E(5) is {_one(wrapped['B'], 'original', 'E5')}, no worse than learned B, "
        f"and wrapped D+B’s E(5) is {_one(wrapped['DB'], 'original', 'E5')}, the same as D+B to three figures. "
        "Built-in symmetry did not hurt accuracy on this test set. It would hurt a model whose eight transformed forecasts pointed different ways."
    )
    pdf.h3("Q2.16")
    pdf.body(
        "On video 9168 the disagreement is gone and the forecast is not repaired. "
        "In Figure 3, row 4 kept two blurred digits and row 3 lost one. "
        "In Figure 6(a) those two rows match: both are the same coarse, blocky field. "
        "That match is the group average. It is why C_R is a floating-point zero. "
        "It is not a better account of the bounce. The upright forecast in row 1 is blocky as well, "
        "which lines up with the original E(5) rising from 0.0223 for A to about 0.030 for D+B. "
        "The part of the failure that was a broken symmetry was fixed by the wrapper, and by D+B’s symmetry augmentation before that. "
        "The part that was a lost digit was not turned back into two sharp trajectories. "
        "Figure 6(b) is the same story at speed. The error at 3× is lower than model A’s, and the frames are still smears rather than a 1 and a 4 at the true positions."
    )
    pdf.h3("Q2.17")
    verdict = (
        "The final model has not learned a motion rule that stands without help. "
        "Model C, with A’s 8-pixel reach, cuts E(2) at 2× from 0.0449 to 0.0288 and at 3× from 0.0523 to 0.0336, "
        "so the speed collapse was motion the training clips never showed. "
        "D’s 16-pixel reach does not beat C, so reach was not the binding limit. "
        "Mirror and rotation consistency are exact only because the final forecaster averages all eight symmetries. That is imposed. "
        "Model B learns a smaller gap, not a zero gap. "
        "A skeptic should ask for a change that neither the training distribution nor the wrapper covers. "
        "The test it has not passed is a rotation by an angle other than a multiple of 90°, or a speed above 3×. "
        "Figure 6 is the other warning: the error can fall while the digits turn into blocks."
    )
    pdf.body(verdict)


if __name__ == "__main__":
    main()
