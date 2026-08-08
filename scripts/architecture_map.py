"""
Perspective architecture map generator.

This script scans the known workspace structure and renders a technical
overview of the system as a four-block flowchart:

1. Frontend / UI
2. Backend / API
3. ML / Inference
4. Data Pipeline

It also prints a categorized file inventory so the diagram is backed by the
actual repository layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "generated" / "perspective_architecture_map.png"


CATEGORIES = {
    "Frontend / UI": {
        "color": "#e8f0ff",
        "edge": "#2f5bea",
        "files": [
            "frontend/src/App.js",
            "frontend/src/index.js",
            "frontend/src/services/api.js",
            "frontend/src/components/Header.js",
            "frontend/src/components/InputSection.js",
            "frontend/src/components/ResultsSection.js",
            "frontend/src/components/NewsFeed.js",
            "frontend/src/components/FactCheckSection.js",
            "frontend/src/components/BiasExplanations.js",
            "frontend/src/components/Footer.js",
            "frontend/src/components/ErrorBoundary.js",
            "frontend/public/index.html",
            "frontend/package.json",
        ],
        "responsibilities": [
            "Collects text or URL input from the user",
            "Calls the Flask API through axios/fetch",
            "Renders bias scores, summaries, and live news cards",
            "Tracks UI state such as loading, server health, and active tabs",
        ],
        "entry_points": [
            "React root: frontend/src/index.js",
            "Main orchestrator: frontend/src/App.js",
            "API client: frontend/src/services/api.js",
        ],
    },
    "Backend / API": {
        "color": "#eef8ef",
        "edge": "#1f8a4c",
        "files": [
            "backend/run.py",
            "backend/app/__init__.py",
            "backend/app/config.py",
            "backend/app/routes.py",
            "backend/app/news_feed.py",
            "backend/app/fact_check.py",
            "backend/tests/conftest.py",
            "backend/tests/test_api.py",
        ],
        "responsibilities": [
            "Exposes /api/health, /api/analyze, /api/news/feed, and bias metadata endpoints",
            "Validates requests, enforces rate limits, and applies security headers",
            "Loads the model lazily and falls back to heuristics if needed",
            "Fetches and normalizes live news / article URLs and fact checks",
        ],
        "entry_points": [
            "Process entry: backend/run.py",
            "Flask factory: backend/app/__init__.py",
            "Route controller: backend/app/routes.py",
        ],
    },
    "ML / Inference": {
        "color": "#fff4e8",
        "edge": "#d97706",
        "files": [
            "model/config/model_config.py",
            "model/src/bert_classifier.py",
            "model/src/dataset.py",
            "model/src/train.py",
            "model/src/inference.py",
            "model/checkpoints/perspective_model/*",
            "model/__init__.py",
        ],
        "responsibilities": [
            "Tokenizes article text to 512 tokens using bert-base-uncased",
            "Runs multi-label classification with 7 independent sigmoid outputs",
            "Converts logits to probabilities and thresholds them into bias labels",
            "Loads local checkpoints and supports both HuggingFace and legacy formats",
        ],
        "entry_points": [
            "Inference runtime: model/src/inference.py",
            "Model definition: model/src/bert_classifier.py",
            "Training config: model/config/model_config.py",
        ],
    },
    "Data Pipeline": {
        "color": "#f8eef8",
        "edge": "#8b5cf6",
        "files": [
            "data/schema.py",
            "data/raw/news_articles.csv",
            "data/processed/labeled_articles.csv",
            "data/splits/train.csv",
            "data/splits/val.csv",
            "data/splits/test.csv",
            "scripts/data_collection/gdelt_collector.py",
            "scripts/data_collection/indian_news_collector.py",
            "scripts/data_collection/news_api_collector.py",
            "scripts/labeling/gemini_labeler.py",
            "scripts/labeling/llm_labeler.py",
            "scripts/preprocessing/clean_data.py",
            "scripts/preprocessing/split_dataset.py",
            "notebooks/colab_training.md",
        ],
        "responsibilities": [
            "Collects raw Indian news articles from GDELT, RSS, or NewsAPI",
            "Cleans text, deduplicates rows, and splits train/val/test sets",
            "Generates labels with Gemini-assisted workflows and human verification",
            "Produces checkpoints that the ML layer consumes during inference",
        ],
        "entry_points": [
            "Collection: scripts/data_collection/gdelt_collector.py",
            "Labeling: scripts/labeling/gemini_labeler.py",
            "Preprocessing: scripts/preprocessing/clean_data.py",
        ],
    },
}


CONNECTIONS = [
    {
        "source": "Frontend / UI",
        "target": "Backend / API",
        "label": "POST /api/analyze, GET /api/health, GET /api/news/feed",
        "detail": "User input becomes JSON payload; React waits for a structured response.",
    },
    {
        "source": "Backend / API",
        "target": "ML / Inference",
        "label": "Lazy BiasPredictor load + tokenization + inference",
        "detail": "routes.py calls model/src/inference.py and falls back to heuristics if the model is not ready.",
    },
    {
        "source": "Data Pipeline",
        "target": "ML / Inference",
        "label": "train.csv, val.csv -> checkpoint artifacts",
        "detail": "Training scripts produce the checkpoint directory that inference loads at runtime.",
    },
    {
        "source": "Data Pipeline",
        "target": "Backend / API",
        "label": "Live news feed + fact-check context",
        "detail": "RSS/GDELT-derived article metadata powers news_feed.py and URL fallback text.",
    },
]


SHARED_AND_INFRASTRUCTURE = [
    "README.md", "docs/*.md", "Dockerfile", "Dockerfile.dev", "docker-compose.yml", "requirements.txt"
]


def _wrap_lines(items: list[str], width: int = 44) -> str:
    return "\n".join(fill(item, width=width) for item in items)


def _category_block_text(category: str) -> str:
    data = CATEGORIES[category]
    lines = []
    lines.append("Files:")
    lines.extend(f"- {item}" for item in data["files"][:7])
    if len(data["files"]) > 7:
        lines.append(f"- ... (+{len(data['files']) - 7} more)")
    lines.append("")
    lines.append("Responsibilities:")
    lines.extend(f"- {item}" for item in data["responsibilities"])
    lines.append("")
    lines.append("Entry points:")
    lines.extend(f"- {item}" for item in data["entry_points"])
    return _wrap_lines(lines, width=46)


def _print_inventory() -> None:
    print("Perspective workspace inventory\n")
    for category, data in CATEGORIES.items():
        print(f"[{category}]")
        for path in data["files"]:
            print(f"  - {path}")
        print()

    print("[Shared / infrastructure]")
    for item in SHARED_AND_INFRASTRUCTURE:
        print(f"  - {item}")
    print()


def _draw_box(ax, center_x, center_y, width, height, facecolor, edgecolor, title, body, title_color="#111827"):
    left = center_x - width / 2
    bottom = center_y - height / 2
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=2.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(center_x, center_y + height * 0.33, title, ha="center", va="center", fontsize=16, fontweight="bold", color=title_color)
    ax.text(center_x, center_y - height * 0.02, body, ha="center", va="center", fontsize=9.2, color="#1f2937", family="monospace")


def _connect(ax, source_xy, target_xy, color, label, detail):
    arrow = FancyArrowPatch(
        source_xy,
        target_xy,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color=color,
        connectionstyle="arc3,rad=0.0",
        zorder=3,
    )
    ax.add_patch(arrow)

    mid_x = (source_xy[0] + target_xy[0]) / 2
    mid_y = (source_xy[1] + target_xy[1]) / 2
    ax.text(mid_x, mid_y + 0.03, label, ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=color)
    ax.text(mid_x, mid_y - 0.02, fill(detail, width=42), ha="center", va="top", fontsize=8.2, color="#374151")


def build_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes([0.03, 0.03, 0.94, 0.9])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Perspective System Architecture Map", fontsize=22, fontweight="bold", y=0.975)
    ax.text(
        0.5,
        0.955,
        "Four-block view of the full-stack MVP, with control flow, training flow, and runtime dependencies",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    positions = {
        "Frontend / UI": (0.25, 0.70),
        "Backend / API": (0.75, 0.70),
        "Data Pipeline": (0.25, 0.28),
        "ML / Inference": (0.75, 0.28),
    }

    for category, (cx, cy) in positions.items():
        data = CATEGORIES[category]
        _draw_box(
            ax,
            cx,
            cy,
            0.38,
            0.28,
            data["color"],
            data["edge"],
            category,
            _category_block_text(category),
        )

    # Connect the blocks to reflect both training-time and runtime dependencies.
    _connect(ax, (0.44, 0.70), (0.56, 0.70), CATEGORIES["Backend / API"]["edge"], CONNECTIONS[0]["label"], CONNECTIONS[0]["detail"])
    _connect(ax, (0.75, 0.56), (0.75, 0.42), CATEGORIES["ML / Inference"]["edge"], CONNECTIONS[1]["label"], CONNECTIONS[1]["detail"])
    _connect(ax, (0.25, 0.14), (0.75, 0.14), CATEGORIES["ML / Inference"]["edge"], CONNECTIONS[2]["label"], CONNECTIONS[2]["detail"])
    _connect(ax, (0.44, 0.28), (0.56, 0.56), CATEGORIES["Data Pipeline"]["edge"], CONNECTIONS[3]["label"], CONNECTIONS[3]["detail"])

    # Add a small legend for the system boundary.
    legend_text = (
        "Runtime path: user -> React -> Flask -> model -> JSON response\n"
        "Training path: raw articles -> cleaned/labelled data -> splits -> checkpoint\n"
        "Operational safety: rate limiting, URL sanitization, fact-checking, fallback heuristics"
    )
    legend = FancyBboxPatch((0.17, 0.86), 0.66, 0.08, boxstyle="round,pad=0.015", facecolor="#f9fafb", edgecolor="#d1d5db")
    ax.add_patch(legend)
    ax.text(0.5, 0.895, legend_text, ha="center", va="center", fontsize=9.5, color="#111827")

    footer = (
        f"Shared / infra files: {', '.join(SHARED_AND_INFRASTRUCTURE)}"
    )
    ax.text(0.5, 0.03, fill(footer, width=160), ha="center", va="center", fontsize=8.8, color="#6b7280")

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a visual architecture map for Perspective.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the PNG output file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    _print_inventory()
    build_figure(args.output)
    print(f"Architecture map written to: {args.output}")


if __name__ == "__main__":
    main()