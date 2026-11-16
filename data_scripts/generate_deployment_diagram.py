"""Generate publication-quality system deployment diagram for Deliverable 3."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_deployment_diagram(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.axis("off")

    c_border = "#2C3E50"
    c_text = "#2C3E50"

    # Title
    ax.text(0.5, 0.97, "AI Demand Forecasting & Inventory Decision Support: System Deployment Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color=c_text)
    ax.text(0.5, 0.935, "Phase 5 Full-Stack Implementation: React 18 SPA, FastAPI Microservice, & Quantile Decision Engine",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#555")

    # Layer 1: Client Tier (React 18 Frontend)
    rect_client = patches.FancyBboxPatch((0.03, 0.53), 0.94, 0.36, boxstyle="round,pad=0.02",
                                         facecolor="#F4F9F9", edgecolor="#3498DB", linewidth=2)
    ax.add_patch(rect_client)
    ax.text(0.06, 0.86, "CLIENT TIER: React 18 Single-Page Application (Vite Dev / Nginx Production)",
            ha="left", va="center", fontsize=12, fontweight="bold", color="#1B4F72")

    # Client components
    ui_boxes = [
        ("Executive Dashboard\n(FR-09, FR-10)", "KPI cards, status filters,\nilluminated P10-P90 bands,\nmedian trendlines", 0.06),
        ("Inventory Cockpit\n(FR-07, FR-08, FR-12)", "Interactive sliders (L, R, S, I, z),\nreal-time safety stock SS,\nAccept/Reject/Adjust actions", 0.29),
        ("CSV Ingestion Gate\n(FR-01, FR-02)", "Drag-and-drop validation,\nschema integrity checks,\nimmediate batch forecasts", 0.52),
        ("Model Registry & Audit\n(FR-11, FR-13)", "Holdout metrics, ROC/PR curves,\nruntime model switching,\nmanager decision logging", 0.75),
    ]

    for title, desc, x_pos in ui_boxes:
        box = patches.FancyBboxPatch((x_pos, 0.56), 0.19, 0.26, boxstyle="round,pad=0.01",
                                     facecolor="#FFFFFF", edgecolor="#85C1E9", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x_pos + 0.095, 0.76, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1B4F72", multialignment="center")
        ax.text(x_pos + 0.095, 0.64, desc, ha="center", va="center", fontsize=8, color="#444", multialignment="center")

    # Layer 2: API & Gateway Tier
    rect_gw = patches.FancyBboxPatch((0.03, 0.35), 0.94, 0.12, boxstyle="round,pad=0.015",
                                     facecolor="#EBF5FB", edgecolor="#2980B9", linewidth=1.5)
    ax.add_patch(rect_gw)
    ax.text(0.5, 0.435, "API GATEWAY & REST CONTRACT TIER: FastAPI (Uvicorn ASGI, Port 8000)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1A5276")
    ax.text(0.5, 0.385, "Endpoints: GET /api/health  |  GET /api/series & /series/{id}  |  POST /api/reorder/calculate  |  POST /api/upload  |  POST /api/models/switch",
            ha="center", va="center", fontsize=8.5, color="#2C3E50", fontfamily="monospace")

    # Arrows between Client and Gateway
    ax.annotate("", xy=(0.5, 0.47), xytext=(0.5, 0.53),
                arrowprops=dict(arrowstyle="<->", lw=2, color="#2980B9"))
    ax.text(0.51, 0.50, "HTTP / JSON REST Calls via api/client.js", fontsize=8.5, color="#1B4F72", fontweight="bold")

    # Layer 3: Backend Business Logic & Model Storage Tier
    rect_backend = patches.FancyBboxPatch((0.03, 0.04), 0.94, 0.26, boxstyle="round,pad=0.02",
                                          facecolor="#FAF9F6", edgecolor="#27AE60", linewidth=2)
    ax.add_patch(rect_backend)
    ax.text(0.06, 0.27, "BUSINESS LOGIC, INFERENCE & STORAGE TIER (Python 3.12+)",
            ha="left", va="center", fontsize=12, fontweight="bold", color="#145A32")

    backend_boxes = [
        ("Replenishment Decision Engine", "SS = z * sigma_d * sqrt(T)\nQ = max(0, ceil(D_T + SS - S - I))\nAlert: S + I < P90_L", 0.06, "#EAFAF1", "#2ECC71"),
        ("Quantile Inference Engine", "Active LightGBM & PyTorch Models\nMulti-horizon direct forecasting\nMonotonic non-crossing sorting", 0.37, "#FEF9E7", "#F1C40F"),
        ("Storage & Model Checkpoints", "models/*.txt (LightGBM Boosters)\nmodels/*.pt (PyTorch Checkpoints)\ndata/processed/*.parquet (History)", 0.68, "#FDEDEC", "#E74C3C"),
    ]

    for title, desc, x_pos, fcol, ecol in backend_boxes:
        box = patches.FancyBboxPatch((x_pos, 0.07), 0.26, 0.17, boxstyle="round,pad=0.01",
                                     facecolor=fcol, edgecolor=ecol, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x_pos + 0.13, 0.20, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2C3E50", multialignment="center")
        ax.text(x_pos + 0.13, 0.125, desc, ha="center", va="center", fontsize=8, color="#333", multialignment="center")

    # Arrows between Gateway and Backend
    ax.annotate("", xy=(0.19, 0.24), xytext=(0.19, 0.35), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#27AE60"))
    ax.annotate("", xy=(0.50, 0.24), xytext=(0.50, 0.35), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#27AE60"))
    ax.annotate("", xy=(0.81, 0.24), xytext=(0.81, 0.35), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#27AE60"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "report" / "figures" / "system_deployment_diagram.png"
    generate_deployment_diagram(out)
    print(f"Generated {out}")
