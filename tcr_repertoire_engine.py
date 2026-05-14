import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import shutil, os

# ── Parameters ──────────────────────────────────────────────────────────────
N_DONORS    = 50
N_CLONO     = 10000   # clonotypes per donor

# ── V/J gene definitions ─────────────────────────────────────────────────────
TRBV_genes = [f'TRBV{i}-{j}' for i in range(1, 31) for j in [1, 2] if i <= 20][:30]
TRBJ_genes = [f'TRBJ{i}-{j}' for i in range(1, 3) for j in range(1, 8)][:13]

N_V = len(TRBV_genes)
N_J = len(TRBJ_genes)

# ── Simulate repertoire ───────────────────────────────────────────────────────
# V gene usage frequencies (non-uniform)
v_freq_base = np.random.dirichlet(np.ones(N_V) * 0.5)
j_freq_base = np.random.dirichlet(np.ones(N_J) * 1.0)

# Per-donor V/J usage
v_usage = np.zeros((N_DONORS, N_V))
j_usage = np.zeros((N_DONORS, N_J))
cdr3_lengths = np.zeros((N_DONORS, N_CLONO), dtype=int)
clone_freqs  = np.zeros((N_DONORS, N_CLONO))

for d in range(N_DONORS):
    v_freq_d = np.random.dirichlet(v_freq_base * 10)
    j_freq_d = np.random.dirichlet(j_freq_base * 10)
    v_usage[d] = v_freq_d
    j_usage[d] = j_freq_d
    # CDR3 lengths: normal around 14 aa
    cdr3_lengths[d] = np.random.normal(14, 2.5, N_CLONO).astype(int).clip(8, 22)
    # Clone frequencies: power law (Zipf-like)
    raw = np.random.exponential(1, N_CLONO)
    raw = raw / raw.sum()
    clone_freqs[d] = np.sort(raw)[::-1]

# ── Clonal diversity metrics ──────────────────────────────────────────────────
shannon_entropy = np.zeros(N_DONORS)
simpson_index   = np.zeros(N_DONORS)
d50_index       = np.zeros(N_DONORS)
top10_fraction  = np.zeros(N_DONORS)

for d in range(N_DONORS):
    p = clone_freqs[d]
    p = p[p > 0]
    shannon_entropy[d] = -np.sum(p * np.log(p + 1e-12))
    simpson_index[d]   = 1 - np.sum(p**2)
    # D50: number of clones comprising top 50%
    cumsum = np.cumsum(p)
    d50_index[d] = np.searchsorted(cumsum, 0.5) + 1
    top10_fraction[d] = p[:10].sum()

# ── Convergent recombination ──────────────────────────────────────────────────
# Simulate CDR3 sequences (as integer hashes)
n_public = 200   # public clonotypes shared across donors
public_cdr3 = np.random.randint(0, 1000000, n_public)
# Each donor has some public clonotypes
donor_public_count = np.random.poisson(20, N_DONORS)

# ── Disease-associated clonotypes ─────────────────────────────────────────────
# COVID vs cancer vs healthy
n_healthy = 20; n_covid = 15; n_cancer = 15
disease_labels = (['Healthy']*n_healthy + ['COVID']*n_covid + ['Cancer']*n_cancer)
# Disease-specific expansion
covid_expansion  = top10_fraction[n_healthy:n_healthy+n_covid] * 1.5
cancer_expansion = top10_fraction[n_healthy+n_covid:] * 1.8
healthy_expansion = top10_fraction[:n_healthy]

# ── VJ pairing matrix ─────────────────────────────────────────────────────────
vj_matrix = np.zeros((N_V, N_J))
for d in range(N_DONORS):
    for v in range(N_V):
        for j in range(N_J):
            vj_matrix[v, j] += v_usage[d, v] * j_usage[d, j]
vj_matrix /= N_DONORS

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('TCR Repertoire Engine — Dashboard', color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, color='#8b949e')
    ax.set_ylabel(ylabel, color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1 — V gene usage heatmap
ax = axes[0,0]
im = ax.imshow(v_usage[:20, :15].T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(15))
ax.set_yticklabels(TRBV_genes[:15], color='white', fontsize=6)
ax.set_xlabel('Donor', color='#8b949e')
plt.colorbar(im, ax=ax, label='Usage Frequency')
style_ax(ax, 'TRBV Gene Usage Heatmap', 'Donor', 'TRBV Gene')

# Panel 2 — CDR3 length distribution
ax = axes[0,1]
all_cdr3 = cdr3_lengths.flatten()
ax.hist(all_cdr3, bins=np.arange(7.5, 23.5), color='#58a6ff', edgecolor='#0d1117', alpha=0.85, density=True)
ax.axvline(all_cdr3.mean(), color='#f78166', lw=2, ls='--',
           label=f'Mean={all_cdr3.mean():.1f} aa')
style_ax(ax, 'CDR3 Length Distribution', 'CDR3 Length (aa)', 'Density')
ax.legend(fontsize=9, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 3 — Clonal diversity comparison
ax = axes[0,2]
disease_colors = {'Healthy': '#3fb950', 'COVID': '#f78166', 'Cancer': '#d2a8ff'}
groups = [('Healthy', shannon_entropy[:n_healthy]),
          ('COVID',   shannon_entropy[n_healthy:n_healthy+n_covid]),
          ('Cancer',  shannon_entropy[n_healthy+n_covid:])]
positions = [1, 2, 3]
for pos, (label, data) in zip(positions, groups):
    bp = ax.boxplot(data, positions=[pos], patch_artist=True, widths=0.5,
                    medianprops={'color': 'white', 'lw': 2})
    bp['boxes'][0].set_facecolor(disease_colors[label])
    bp['boxes'][0].set_alpha(0.7)
    for el in ['whiskers', 'caps', 'fliers']:
        for item in bp[el]:
            item.set_color('#8b949e')
ax.set_xticks(positions)
ax.set_xticklabels(['Healthy', 'COVID', 'Cancer'], color='white')
style_ax(ax, 'Clonal Diversity (Shannon Entropy)', 'Group', 'Shannon Entropy')

# Panel 4 — Expansion index
ax = axes[1,0]
ax.scatter(range(n_healthy), healthy_expansion, c='#3fb950', s=50, alpha=0.8, label='Healthy')
ax.scatter(range(n_healthy, n_healthy+n_covid), covid_expansion, c='#f78166', s=50, alpha=0.8, label='COVID')
ax.scatter(range(n_healthy+n_covid, N_DONORS), cancer_expansion, c='#d2a8ff', s=50, alpha=0.8, label='Cancer')
style_ax(ax, 'Clonal Expansion (Top 10 Clones)', 'Donor', 'Top 10 Clone Fraction')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 5 — Convergent recombination
ax = axes[1,1]
ax.bar(range(N_DONORS), donor_public_count, color='#ffa657', edgecolor='#0d1117', alpha=0.85)
ax.axhline(donor_public_count.mean(), color='white', lw=2, ls='--',
           label=f'Mean={donor_public_count.mean():.1f}')
style_ax(ax, 'Convergent Recombination (Public Clonotypes)', 'Donor', 'Public Clonotype Count')
ax.legend(fontsize=9, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 6 — J gene usage
ax = axes[1,2]
j_mean = j_usage.mean(axis=0)
j_se   = j_usage.std(axis=0) / np.sqrt(N_DONORS)
ax.bar(range(N_J), j_mean, yerr=j_se, color='#3fb950', capsize=3, edgecolor='#0d1117', alpha=0.85)
ax.set_xticks(range(N_J))
ax.set_xticklabels(TRBJ_genes, rotation=45, ha='right', color='white', fontsize=7)
style_ax(ax, 'TRBJ Gene Usage', 'TRBJ Gene', 'Mean Usage Frequency')

# Panel 7 — VJ pairing
ax = axes[2,0]
im2 = ax.imshow(vj_matrix[:15, :], aspect='auto', cmap='Blues', interpolation='nearest')
ax.set_yticks(range(15))
ax.set_yticklabels(TRBV_genes[:15], color='white', fontsize=6)
ax.set_xticks(range(N_J))
ax.set_xticklabels(TRBJ_genes, rotation=45, ha='right', color='white', fontsize=6)
plt.colorbar(im2, ax=ax, label='Pairing Frequency')
style_ax(ax, 'VJ Pairing Matrix', 'TRBJ', 'TRBV')

# Panel 8 — Disease clonotype enrichment
ax = axes[2,1]
d50_healthy = d50_index[:n_healthy]
d50_covid   = d50_index[n_healthy:n_healthy+n_covid]
d50_cancer  = d50_index[n_healthy+n_covid:]
groups_d50 = [d50_healthy, d50_covid, d50_cancer]
labels_d50 = ['Healthy', 'COVID', 'Cancer']
colors_d50 = ['#3fb950', '#f78166', '#d2a8ff']
for i, (data, label, color) in enumerate(zip(groups_d50, labels_d50, colors_d50)):
    ax.bar(i, data.mean(), yerr=data.std(), color=color, capsize=5,
           edgecolor='#0d1117', alpha=0.85, label=label)
ax.set_xticks(range(3))
ax.set_xticklabels(labels_d50, color='white')
style_ax(ax, 'Disease Clonotype Enrichment (D50)', 'Group', 'D50 Index')

# Panel 9 — Summary
ax = axes[2,2]
ax.axis('off')
style_ax(ax, 'Summary Statistics')
summary = [
    f'Donors: {N_DONORS}',
    f'Clonotypes/donor: {N_CLONO:,}',
    f'TRBV genes: {N_V}',
    f'TRBJ genes: {N_J}',
    f'Mean CDR3 length: {all_cdr3.mean():.1f} aa',
    f'Mean Shannon entropy: {shannon_entropy.mean():.2f}',
    f'Mean Simpson index: {simpson_index.mean():.4f}',
    f'Mean D50: {d50_index.mean():.0f} clones',
    f'Mean top-10 fraction: {top10_fraction.mean():.3f}',
    f'Public clonotypes: {n_public}',
]
for k, line in enumerate(summary):
    ax.text(0.05, 0.92 - k*0.09, line, transform=ax.transAxes,
            color='#e6edf3', fontsize=10, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = '/mnt/shared-workspace/shared/tcr_repertoire_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'Dashboard saved: {out_png}')

shutil.copy('/workspace/subagents/a29c645f/tcr_repertoire_engine.py',
            '/mnt/shared-workspace/shared/tcr_repertoire_engine.py')

print('\n=== KEY RESULTS: TCRRepertoireEngine ===')
print(f'Donors: {N_DONORS}, Clonotypes/donor: {N_CLONO:,}')
print(f'Mean CDR3 length: {all_cdr3.mean():.1f} aa')
print(f'Mean Shannon entropy: {shannon_entropy.mean():.2f}')
print(f'Mean Simpson index: {simpson_index.mean():.4f}')
print(f'Mean D50: {d50_index.mean():.0f} clones')
print(f'Mean top-10 fraction: {top10_fraction.mean():.3f}')
print(f'COVID expansion vs Healthy: {covid_expansion.mean():.3f} vs {healthy_expansion.mean():.3f}')
