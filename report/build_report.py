"""Build an academic PDF from Markdown, measured JSON/CSV and portable fonts."""

from html import escape
import json
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
NUMBER = 2 if (ROOT / "dags").exists() else 1
INK = "#163444"
TEAL = "#087f8c"
WIDTH = A4[0] - 100
REPOSITORY = "https://github.com/TrinhDucDuong/" + (
    "ddm501-assignment-1-telco-churn-system-design"
    if NUMBER == 1 else "ddm501-assignment-2-telco-churn-mlops-pipeline"
)
SOURCE_COMMIT = (
    "290f2fd761fea87338166124eb507fe30c162a26"
    if NUMBER == 1 else "627989c424cd765635af5249a4a3819e964be181"
)


def diagram(filename: str, nodes: list, edges: list, title: str, height: float = 4.2) -> None:
    """Draw labeled stages and explicit directed data/control flows."""
    fig, ax = plt.subplots(figsize=(10.8, height))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, height)
    ax.axis("off")
    positions = {}
    for key, x, y, label, detail in nodes:
        positions[key] = (x, y)
        box = FancyBboxPatch(
            (x, y),
            2.7,
            0.76,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="#edf6f6",
            edgecolor=TEAL,
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 1.35, y + 0.49, label, ha="center", va="center", fontsize=10.1, weight="bold", color=INK)
        ax.text(x + 1.35, y + 0.2, detail, ha="center", va="center", fontsize=8, color="#4b606a")
    for first, second in edges:
        x1, y1 = positions[first]
        x2, y2 = positions[second]
        if abs(y1 - y2) < 0.1:
            start, end = (
                ((x1 + 2.75, y1 + 0.38), (x2 - 0.07, y2 + 0.38))
                if x2 > x1
                else ((x1 - 0.06, y1 + 0.38), (x2 + 2.77, y2 + 0.38))
            )
        else:
            start, end = (x1 + 1.35, y1 - 0.07), (x2 + 1.35, y2 + 0.83)
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.35})
    ax.text(0.15, height - 0.16, title, fontsize=11, weight="bold", color=INK, va="top")
    fig.savefig(REPORT / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures() -> None:
    """Generate architecture and comparison figures from the actual project outputs."""
    diagram(
        "architecture.png",
        [
            ("source", 0.2, 2.8, "Source snapshots", "Billing + service + CRM"),
            ("quality", 4.0, 2.8, "Validation / versioning", "Schema + cutoff + lineage"),
            ("train", 7.8, 2.8, "Train and evaluate", "Train-only transforms"),
            ("model", 7.8, 1.3, "Approved model", "Gate + human review"),
            ("score", 4.0, 1.3, "Daily batch / API", "Risk + top-budget ranking"),
            ("crm", 0.2, 1.3, "Agent review / CRM", "Consent + action logging"),
            ("feedback", 0.2, 0.1, "Matured outcomes", "Business + model monitoring"),
        ],
        [
            ("source", "quality"),
            ("quality", "train"),
            ("train", "model"),
            ("model", "score"),
            ("score", "crm"),
            ("crm", "feedback"),
        ],
        "PRODUCTION DESIGN | offline training and daily retention workflow",
        4.2,
    )
    if NUMBER == 2:
        diagram(
            "pipeline.png",
            [
                ("a", 0.2, 3.1, "01 Ingest", "CSV + SHA-256 manifest"),
                ("b", 4.0, 3.1, "02 Validate", "Schema / labels / ranges"),
                ("c", 7.8, 3.1, "03 Prepare", "60 / 20 / 20 split + IDs"),
                ("d", 7.8, 1.75, "04 Train + features", "Fit transforms; 10 MLflow runs"),
                ("e", 4.0, 1.75, "05 Evaluate", "Validation winner; holdout"),
                ("f", 0.2, 1.75, "06 Gated deployment", "Registry + local artifact"),
                ("g", 0.2, 0.4, "07 Score", "Ranked CSV; top-20% flag"),
                ("h", 4.0, 0.4, "Evidence / lineage", "Metrics, hashes, run + version"),
            ],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")],
            "IMPLEMENTED PIPELINE | persisted boundaries and shared stage functions",
            4.5,
        )
        fig, ax = plt.subplots(figsize=(10.8, 2.2))
        ax.axis("off")
        names = ["ingest", "validate", "prepare", "train", "evaluate", "deploy", "score"]
        for i, name in enumerate(names):
            x = i * 1.5
            ax.add_patch(
                FancyBboxPatch(
                    (x, 0.8), 1.25, 0.6, boxstyle="round,pad=.03", facecolor="#edf6f6", edgecolor=TEAL
                )
            )
            ax.text(x + 0.625, 1.1, name, fontsize=9, ha="center", va="center", color=INK)
            if i < 6:
                ax.annotate(
                    "",
                    xy=(x + 1.47, 1.1),
                    xytext=(x + 1.28, 1.1),
                    arrowprops={"arrowstyle": "->", "color": INK},
                )
        ax.text(0, 1.85, "AIRFLOW DAG | telco_churn_training", weight="bold", color=INK, fontsize=11)
        ax.text(
            0,
            0.25,
            "Sunday 02:00 Asia/Ho_Chi_Minh | all_success | 2 retries | max_active_runs=1",
            fontsize=9,
            color=INK,
        )
        ax.set_xlim(-0.1, 10.5)
        ax.set_ylim(0, 2.1)
        fig.savefig(REPORT / "dag.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        df = pd.read_csv(ROOT / "artifacts/experiments.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
        labels = df.name.str[:3]
        palette = [TEAL if name == "E05" else "#8eacb5" for name in labels]
        axes[0].barh(labels, df.average_precision, color=palette)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Validation average precision")
        axes[0].set_xlim(0, 0.75)
        axes[0].axvline(0.60, color="#b66a29", linestyle="--", linewidth=1, label="AP gate")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].barh(labels, df.p95_model_latency_ms, color=palette)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Warm single-row model p95 (ms)")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.15)
        fig.suptitle("Measured comparison | one frozen validation split", color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(REPORT / "experiments.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def measured_section() -> str:
    """Create report tables directly from executed output; never use invented metrics."""
    summary = json.loads((ROOT / "artifacts/summary.json").read_text())
    verification_file = ROOT / "evidence/verification.json"
    evidence = json.loads(verification_file.read_text()) if verification_file.exists() else None
    if NUMBER == 1:
        rows = ["| Metric | Prior baseline (test) | Logistic C=1 (test) |", "|---|---|---|"]
        for metric in [
            "average_precision",
            "roc_auc",
            "brier",
            "f1",
            "precision_at_capacity",
            "recall_at_capacity",
            "lift_at_capacity",
        ]:
            rows.append(
                f"| {metric.replace('_', ' ')} | {summary['baseline_test'][metric]:.4f} | {summary['test'][metric]:.4f} |"
            )
        result = "\n".join(rows)
        result += f"\n\nMeasured training time: {summary['fit_seconds']:.4f} seconds. Warm single-row model p95 latency: {summary['p95_model_latency_ms']:.2f} ms (50 repetitions; excludes HTTP/network). Validation AP: {summary['validation']['average_precision']:.4f}. These are local measurements, not throughput guarantees.\n"
    else:
        df = pd.read_csv(ROOT / "artifacts/experiments.csv")
        rows = [
            "| Trial | AP | AUC | Brier | F1 | Lift@20% | Fit (s) | p95 (ms) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in df.itertuples():
            rows.append(
                f"| {r.name[:3]} | {r.average_precision:.4f} | {r.roc_auc:.4f} | {r.brier:.4f} | {r.f1:.4f} | {r.lift_at_capacity:.3f} | {r.fit_seconds:.3f} | {r.p95_model_latency_ms:.2f} |"
            )
        result = (
            "All ten rows below are measured on the same validation partition. Timing is hardware- and load-dependent.\n\n"
            + "\n".join(rows)
        )
        result += "\n\n{{EXPERIMENT_FIGURE}}\n\n"
        result += f"Selected candidate: {summary['winner']}. Validation AP={summary['validation']['average_precision']:.4f}; validation lift@20%={summary['validation']['lift_at_capacity']:.4f}. The quality gate passes. No model was reselected based on the following holdout table.\n\n"
        result += "| Holdout metric | Measured value |\n|---|---|\n"
        for metric in [
            "average_precision",
            "roc_auc",
            "brier",
            "f1",
            "precision_at_capacity",
            "recall_at_capacity",
            "lift_at_capacity",
        ]:
            result += f"| {metric.replace('_', ' ')} | {summary['test'][metric]:.4f} |\n"
        champion = json.loads((ROOT / "artifacts/champion.json").read_text())
        result += f"\nRegistry version: {champion['version']}; model: {champion['name']}; alias: candidate. The run ID and full lineage are preserved in evidence/champion.json and the local MLflow store.\n"
    t = summary["test"]
    result += f"\nThe holdout contains {summary['split_sizes']['test']:,} customers. The 20% queue contains {t['selected_count']} records. At threshold 0.5, the confusion counts are TN={t['tn']}, FP={t['fp']}, FN={t['fn']}, TP={t['tp']}. Threshold classifications and the top-budget queue are different policies.\n"
    if evidence:
        result += f"\nExecution evidence: Python {evidence['python']}; project-local environment; dependency, lint, test and full-data pipeline commands exited successfully. A real HTTP server returned 200 for valid prediction and 422 for an invalid empty batch. Machine-readable evidence and full test output are in evidence/verification.json and evidence/pytest.txt. Execution timestamp (UTC): {evidence['executed_utc']}.\n"
    airflow = ROOT / "evidence/airflow-summary.json"
    if NUMBER == 2 and airflow.exists():
        a = json.loads(airflow.read_text())
        result += f"\nActual Airflow execution: {a['state']}; {a['successful_tasks']} of 7 tasks completed; {a['scored_rows']} scored rows and {a['contact_rows']} contact flags were persisted from the Docker run. See evidence/airflow.txt and evidence/airflow-summary.json. This validates real DAG execution, not continuous scheduling or HA.\n"
    return result


def register_fonts() -> None:
    for name, file in [
        ("Body", "DejaVuSans.ttf"),
        ("BodyBold", "DejaVuSans-Bold.ttf"),
        ("Mono", "DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(REPORT / "fonts" / file)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold"
    )


class ReportDoc(BaseDocTemplate):
    """Add page furniture, PDF outline and a generated contents page."""

    def __init__(self, filename: Path):
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=50,
            rightMargin=50,
            topMargin=58,
            bottomMargin=48,
            title=f"DDM501 Assignment {NUMBER}",
            author="Trịnh Đức Dương - 25ms13290",
        )
        self.addPageTemplates(
            PageTemplate(
                id="report",
                frames=[
                    Frame(
                        50,
                        48,
                        WIDTH,
                        A4[1] - 106,
                        id="body",
                        leftPadding=0,
                        rightPadding=0,
                        topPadding=0,
                        bottomPadding=0,
                    )
                ],
                onPage=self.page_decoration,
            )
        )

    def page_decoration(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(INK))
        canvas.setFont("Body", 8)
        if doc.page > 1:
            canvas.drawString(50, A4[1] - 32, f"DDM501 / INDIVIDUAL ASSIGNMENT {NUMBER}")
            canvas.setStrokeColor(colors.HexColor("#c6d9de"))
            canvas.line(50, A4[1] - 40, A4[0] - 50, A4[1] - 40)
        canvas.drawString(50, 27, "Trịnh Đức Dương  |  25ms13290")
        canvas.drawRightString(A4[0] - 50, 27, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1":
            title = flowable.getPlainText()
            key = f"section-{self.seq.nextf('section')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0)
            self.notify("TOCEntry", (0, title, self.page, key))


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Mono">\1</font>', text)
    text = re.sub(
        r"\[([^\]]+)\]\((https://[^\s)]+)\)",
        r'<link href="\2" color="#087f8c"><u>\1</u></link>',
        text,
    )
    return text


def build() -> None:
    register_fonts()
    figures()
    text = (
        (REPORT / "report.md").read_text(encoding="utf-8").replace("{{RESULTS_SECTION}}", measured_section())
    )
    (REPORT / "report.generated.md").write_text(text, encoding="utf-8")
    body = ParagraphStyle(
        "Body", fontName="Body", fontSize=9.5, leading=14, textColor=colors.HexColor(INK), spaceAfter=8
    )
    h1 = ParagraphStyle(
        "H1",
        parent=body,
        fontName="BodyBold",
        fontSize=14,
        leading=19,
        spaceBefore=16,
        spaceAfter=9,
        keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=body,
        fontName="BodyBold",
        fontSize=11.2,
        leading=16,
        spaceBefore=11,
        spaceAfter=6,
        keepWithNext=True,
    )
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=11)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.7, leading=10.7, spaceAfter=0)
    code_style = ParagraphStyle(
        "Code",
        fontName="Mono",
        fontSize=7.2,
        leading=10,
        backColor=colors.HexColor("#f1f5f6"),
        borderPadding=8,
        spaceBefore=7,
        spaceAfter=12,
    )
    title = "ML System Design\nDocument" if NUMBER == 1 else "ML Pipeline Design\n& MLOps Analysis"
    story = [
        Spacer(1, 54),
        Paragraph("FPT SCHOOL OF BUSINESS & TECHNOLOGY", small),
        Spacer(1, 26),
        Paragraph(
            f"INDIVIDUAL ASSIGNMENT {NUMBER}",
            ParagraphStyle(
                "Eyebrow", parent=body, fontName="BodyBold", fontSize=12, textColor=colors.HexColor(TEAL)
            ),
        ),
        Spacer(1, 14),
        Paragraph(
            title.replace("\n", "<br/>"),
            ParagraphStyle("Title", parent=body, fontName="BodyBold", fontSize=31, leading=39),
        ),
        Spacer(1, 24),
        Paragraph(
            "Telecom Customer Churn<br/>A reproducible retention-risk system",
            ParagraphStyle("Sub", parent=body, fontSize=17, leading=25),
        ),
        Spacer(1, 48),
        Paragraph(
            "<b>Trịnh Đức Dương</b><br/>Student ID: 25ms13290<br/>Course: DDM501 - AI in DevOps, DataOps, MLOps<br/>Submission: Individual written report<br/>Revised: 6 October 2026",
            body,
        ),
        Spacer(1, 24),
        Paragraph("<b>Companion GitHub repository</b>", small),
        Paragraph(f'<link href="{REPOSITORY}" color="{TEAL}"><u>{REPOSITORY}</u></link>', small),
        Paragraph(
            f'Implementation snapshot: <link href="{REPOSITORY}/tree/{SOURCE_COMMIT}" '
            f'color="{TEAL}"><u>{SOURCE_COMMIT[:7]}</u></link>. '
            "The PDF is self-contained; code, data and execution evidence supplement the report.",
            small,
        ),
        PageBreak(),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC", parent=body, fontSize=10, leading=16, leftIndent=0, firstLineIndent=0, spaceBefore=7
        )
    ]
    story += [Paragraph("Contents", ParagraphStyle("Contents", parent=h1)), Spacer(1, 10), toc, PageBreak()]
    lines = text.splitlines()
    i = 0
    image_map = {
        "{{ARCHITECTURE_FIGURE}}": (
            "architecture.png",
            "Figure 1. Proposed production architecture; feedback informs monitored retraining.",
        ),
        "{{PIPELINE_FIGURE}}": (
            "pipeline.png",
            "Figure 1. Implemented pipeline and persisted data transformations.",
        ),
        "{{DAG_FIGURE}}": ("dag.png", "Figure 3. Executable Airflow task dependencies and schedule."),
        "{{EXPERIMENT_FIGURE}}": (
            "experiments.png",
            "Figure 2. Measured AP and latency; timing varies with host workload.",
        ),
    }
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("# "):
            i += 1
            continue
        if line in image_map:
            name, caption = image_map[line]
            from PIL import Image as PILImage

            with PILImage.open(REPORT / name) as picture:
                width, height = picture.size
            story.append(
                KeepTogether(
                    [
                        Image(str(REPORT / name), width=WIDTH, height=WIDTH * height / width),
                        Paragraph(caption, small),
                        Spacer(1, 8),
                    ]
                )
            )
            i += 1
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), h2))
            i += 1
        elif line.startswith("## "):
            story.extend([CondPageBreak(280), Paragraph(inline(line[3:]), h1)])
            i += 1
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                raw = lines[i]
                block.extend(
                    textwrap.wrap(
                        raw,
                        width=91,
                        replace_whitespace=False,
                        drop_whitespace=False,
                        subsequent_indent="    ",
                    )
                    or [""]
                )
                i += 1
            story.append(Preformatted("\n".join(block), code_style))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [part.strip() for part in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", part) for part in parts):
                    rows.append(parts)
                i += 1
            count = len(rows[0])
            if count == 2:
                widths = [WIDTH * 0.34, WIDTH * 0.66]
            elif count == 3:
                widths = [WIDTH * 0.23, WIDTH * 0.37, WIDTH * 0.40]
            elif count == 4:
                widths = [WIDTH * 0.11, WIDTH * 0.19, WIDTH * 0.38, WIDTH * 0.32]
            else:
                widths = [WIDTH / count] * count
            content = [
                [
                    Paragraph(("<b>" if r == 0 else "") + inline(value) + ("</b>" if r == 0 else ""), cell)
                    for value in row
                ]
                for r, row in enumerate(rows)
            ]
            table = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbecef")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f8")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(TEAL)),
                    ]
                )
            )
            story += [table, Spacer(1, 11)]
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```", "{{")):
                paragraph.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(paragraph)), body))
    output = ROOT / f"DDM501_Assignment{NUMBER}_25ms13290_TrinhDucDuong.pdf"
    ReportDoc(output).multiBuild(story)
    print(output)


if __name__ == "__main__":
    build()
