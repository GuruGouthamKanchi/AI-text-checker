import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Output directory for clean charts
output_dir = r"C:\Users\Goutham\.gemini\antigravity-ide\brain\494f7a94-1825-42e2-95a6-de48a89abf9a\charts"
os.makedirs(output_dir, exist_ok=True)

# Minimalist, crisp IEEE presentation typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#E2E8F0'
plt.rcParams['axes.linewidth'] = 1.0

# Methods & Data
methods = [
    'Raw AI Text',
    'QuillBot',
    'Kirchenbauer 2023',
    'DetectGPT 2023',
    'Sadasivan 2024',
    'AccaHumanize-CL (Ours)'
]

ai_scores = [99.18, 72.40, 61.80, 58.20, 54.10, 28.40]
integrity_scores = [100.0, 42.0, 65.0, 52.0, 48.0, 100.0]
semantic_retention = [100.0, 71.2, 78.4, 80.1, 74.5, 88.4]
corruption_rates = [0.0, 58.0, 35.0, 48.0, 52.0, 0.0]

# Executive Clean Colors
bar_colors = ['#94A3B8', '#94A3B8', '#94A3B8', '#94A3B8', '#94A3B8', '#059669']

# ---------------------------------------------------------
# CHART 1: Clean AI Detection Score Comparison
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

bars = ax.bar(methods, ai_scores, color=bar_colors, width=0.48, zorder=3)

# Human Threshold Line
ax.axhline(y=30.0, color='#E11D48', linestyle='--', linewidth=1.5, zorder=2, label='Human Safety Threshold (<= 30%)')

# Clean Value Annotations (No boxy clutter)
for bar, score in zip(bars, ai_scores):
    yval = bar.get_height()
    color = '#059669' if score <= 30 else '#0F172A'
    fontweight = 'bold' if score <= 30 else 'normal'
    ax.annotate(f'{score:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, yval + 1.8),
                ha='center', va='bottom', fontsize=10.5, fontweight=fontweight, color=color)

ax.set_ylabel('AI Detection Probability (%)', fontsize=11, fontweight='bold', color='#0F172A', labelpad=8)
ax.set_title('AI Detection Score Comparison Across Methods', fontsize=13, fontweight='bold', color='#0F172A', pad=15)
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')

plt.xticks(rotation=15, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
plt.yticks(fontsize=9.5, color='#475569')
plt.legend(loc='upper right', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clean_chart1_ai_scores.png'))
plt.close()


# ---------------------------------------------------------
# CHART 2: Clean Side-by-Side LaTeX Integrity vs Corruption
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

x = np.arange(len(methods))
width = 0.35

rects1 = ax.bar(x - width/2, integrity_scores, width, label='LaTeX Structural Integrity (%)', color='#059669', zorder=3)
rects2 = ax.bar(x + width/2, corruption_rates, width, label='Formatting Corruption Rate (%)', color='#E11D48', zorder=3)

ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold', color='#0F172A', labelpad=8)
ax.set_title('LaTeX Structural Preservation vs. Formatting Corruption', fontsize=13, fontweight='bold', color='#0F172A', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h + 1.5),
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#059669')

for rect in rects2:
    h = rect.get_height()
    if h > 0:
        ax.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h + 1.5),
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#E11D48')

plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clean_chart2_integrity.png'))
plt.close()


# ---------------------------------------------------------
# CHART 3: Clean Semantic Similarity Retention Bar Chart
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

bars3 = ax.bar(methods, semantic_retention, color=bar_colors, width=0.48, zorder=3)
ax.axhline(y=85.0, color='#059669', linestyle='--', linewidth=1.5, zorder=2, label='SciBERT Target Threshold (>= 85%)')

for bar, score in zip(bars3, semantic_retention):
    yval = bar.get_height()
    color = '#059669' if score >= 85 else '#0F172A'
    fontweight = 'bold' if score >= 85 else 'normal'
    ax.annotate(f'{score:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, yval + 1.8),
                ha='center', va='bottom', fontsize=10.5, fontweight=fontweight, color=color)

ax.set_ylabel('SciBERT Cosine Similarity (%)', fontsize=11, fontweight='bold', color='#0F172A', labelpad=8)
ax.set_title('Semantic Similarity Retention (SciBERT Cosine)', fontsize=13, fontweight='bold', color='#0F172A', pad=15)
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')

plt.xticks(rotation=15, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
plt.yticks(fontsize=9.5, color='#475569')
plt.legend(loc='lower right', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clean_chart3_semantic.png'))
plt.close()


# ---------------------------------------------------------
# CHART 4: Clean Convergence Trajectory Line Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

steps = ['Initial Draft\n(t=0)', 'Pass 1\n(Neural)', 'Pass 2\n(Rule Hybrid)', 'Pass 3\n(Converged)']
scores_doc = [99.18, 57.17, 37.17, 28.40]

ax.plot(steps, scores_doc, marker='o', markersize=8, linewidth=2.5, color='#059669', zorder=3, label='AccaHumanize-CL Mean AI Score')
ax.axhline(y=30.0, color='#E11D48', linestyle='--', linewidth=1.5, zorder=2, label='Human Safety Threshold (30%)')

for i, score in enumerate(scores_doc):
    ax.annotate(f'{score:.1f}%', xy=(i, score + 3), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#059669')

ax.set_ylabel('AI Detection Score (%)', fontsize=11, fontweight='bold', color='#0F172A', labelpad=8)
ax.set_title('Iterative Critic-Generator Closed-Loop Convergence Trajectory', fontsize=13, fontweight='bold', color='#0F172A', pad=15)
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')

plt.xticks(fontsize=9.5, fontweight='bold', color='#334155')
plt.yticks(fontsize=9.5, color='#475569')
plt.legend(loc='upper right', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clean_chart4_convergence.png'))
plt.close()

print("Successfully generated all 4 ultra-clean presentation charts!")
