"""Generate publication-quality architecture diagram for QuantileMLP and LightGBM."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')

    c_input = '#E8F4F8'
    c_embed = '#D1E8E2'
    c_backbone = '#F9F1E6'
    c_head = '#FDE2E4'
    c_lgbm = '#E2ECE9'
    c_border = '#2C3E50'
    c_text = '#2C3E50'

    # Title
    ax.text(0.5, 0.97, 'AI Demand Forecasting: Model Architecture Overview', 
            ha='center', va='center', fontsize=16, fontweight='bold', color=c_text)
    ax.text(0.5, 0.935, 'Quantile Multi-Layer Perceptron (QuantileMLP) & LightGBM Multi-Quantile Ensemble', 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#555')

    # Track 1: QuantileMLP Box (left & middle)
    rect_mlp = patches.FancyBboxPatch((0.03, 0.05), 0.62, 0.84, boxstyle='round,pad=0.02', 
                                       facecolor='#FAFAFA', edgecolor='#4A90E2', linewidth=2)
    ax.add_patch(rect_mlp)
    ax.text(0.34, 0.86, 'Track 1: PyTorch QuantileMLP Architecture', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#1B4F72')

    # Inputs
    rect_cat = patches.FancyBboxPatch((0.06, 0.68), 0.25, 0.13, boxstyle='round,pad=0.01', 
                                      facecolor=c_input, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(rect_cat)
    ax.text(0.185, 0.77, 'Categorical Features', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.185, 0.715, 'Store, Assortment, StoreType,\nPromoInterval, StateHoliday', 
            ha='center', va='center', fontsize=8, color='#444')

    rect_num = patches.FancyBboxPatch((0.36, 0.68), 0.25, 0.13, boxstyle='round,pad=0.01', 
                                      facecolor=c_input, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(rect_num)
    ax.text(0.485, 0.77, 'Continuous Features (t - h)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.485, 0.715, 'Lags (7, 14, 28), Rolling Mean/Std,\nCyclical Sin/Cos, StandardScaler', 
            ha='center', va='center', fontsize=8, color='#444')

    # Embeddings
    rect_emb = patches.FancyBboxPatch((0.06, 0.50), 0.25, 0.12, boxstyle='round,pad=0.01', 
                                      facecolor=c_embed, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(rect_emb)
    ax.text(0.185, 0.575, 'Entity Embeddings', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.185, 0.525, 'd_e = min(32, max(4, |C| // 2))\nTrainable dense vectors', 
            ha='center', va='center', fontsize=8, color='#333')

    # Concat
    rect_concat = patches.FancyBboxPatch((0.15, 0.38), 0.38, 0.07, boxstyle='round,pad=0.01', 
                                        facecolor='#FFEAA7', edgecolor=c_border, linewidth=1.2)
    ax.add_patch(rect_concat)
    ax.text(0.34, 0.415, 'Feature Concatenation: [ e_cat || x_num ]', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#2D3436')

    # Hidden Backbone
    rect_bb = patches.FancyBboxPatch((0.15, 0.22), 0.38, 0.11, boxstyle='round,pad=0.01', 
                                     facecolor=c_backbone, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(rect_bb)
    ax.text(0.34, 0.29, 'Dense Backbone with Dropout (p = 0.2)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.34, 0.245, 'Linear(256) -> ReLU -> Dropout -> Linear(128) -> ReLU -> Linear(64)', 
            ha='center', va='center', fontsize=8, color='#444')

    # Quantile Head
    rect_qh = patches.FancyBboxPatch((0.08, 0.08), 0.52, 0.09, boxstyle='round,pad=0.01', 
                                     facecolor=c_head, edgecolor='#C0392B', linewidth=1.5)
    ax.add_patch(rect_qh)
    ax.text(0.34, 0.135, 'Monotonic Non-Crossing Quantile Head', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#922B21')
    ax.text(0.34, 0.095, 'P50 = z_0   |   P10 = z_0 - softplus(z_1)   |   P90 = z_0 + softplus(z_2)   =>  P10 <= P50 <= P90', 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='#78281F')

    # Arrows in MLP
    ax.annotate('', xy=(0.185, 0.62), xytext=(0.185, 0.68), arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))
    ax.annotate('', xy=(0.25, 0.45), xytext=(0.185, 0.50), arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))
    ax.annotate('', xy=(0.43, 0.45), xytext=(0.485, 0.68), arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))
    ax.annotate('', xy=(0.34, 0.33), xytext=(0.34, 0.38), arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))
    ax.annotate('', xy=(0.34, 0.17), xytext=(0.34, 0.22), arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))

    # Track 2: LightGBM Box (right)
    rect_lgbm = patches.FancyBboxPatch((0.68, 0.05), 0.29, 0.84, boxstyle='round,pad=0.02', 
                                        facecolor='#FAFAFA', edgecolor='#27AE60', linewidth=2)
    ax.add_patch(rect_lgbm)
    ax.text(0.825, 0.86, 'Track 2: LightGBM Quantile Ensemble', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#196F3D')

    y_pos = [0.68, 0.48, 0.28]
    quant_labels = [('Estimator 1 (tau = 0.10)', 'Lower Bound P10 Regressor'), 
                    ('Estimator 2 (tau = 0.50)', 'Median Demand P50 Regressor'), 
                    ('Estimator 3 (tau = 0.90)', 'Upper Bound P90 Regressor')]

    for (q_title, q_sub), yp in zip(quant_labels, y_pos):
        rect_tree = patches.FancyBboxPatch((0.70, yp), 0.25, 0.13, boxstyle='round,pad=0.01', 
                                           facecolor=c_lgbm, edgecolor='#27AE60', linewidth=1.2)
        ax.add_patch(rect_tree)
        ax.text(0.825, yp + 0.08, q_title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#145A32')
        ax.text(0.825, yp + 0.035, q_sub + '\nLeaf-wise split, early stopping', ha='center', va='center', fontsize=7.5, color='#333')

    rect_sort = patches.FancyBboxPatch((0.70, 0.08), 0.25, 0.09, boxstyle='round,pad=0.01', 
                                       facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=1.5)
    ax.add_patch(rect_sort)
    ax.text(0.825, 0.135, 'Monotonic Sorting & Inversion', ha='center', va='center', 
            fontsize=9.5, fontweight='bold', color='#1E8449')
    ax.text(0.825, 0.095, 'sort([P10, P50, P90], axis=1)\nexpm1() & closed-day zero mask', 
            ha='center', va='center', fontsize=7.5, color='#145A32')

    for yp in y_pos:
        ax.annotate('', xy=(0.825, 0.17), xytext=(0.825, yp), arrowprops=dict(arrowstyle='->', lw=1.2, color='#27AE60'))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    generate_diagram(root / 'report' / 'figures' / 'architecture_diagram.png')

