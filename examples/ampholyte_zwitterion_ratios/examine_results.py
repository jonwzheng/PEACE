import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

# Load benchmark results
df = pd.read_csv('results/f_zwit_benchmark/benchmark_results.csv')

# Prepare plot
fig, ax = plt.subplots(figsize=(13,6))

# Make a scatterplot, colored by source
scatter = ax.scatter(
    df['experimental_f_zwit'], 
    df['predicted_f_zwit'], 
    c=pd.Categorical(df['source']).codes, 
    cmap='tab10',
    label=df['source']
)

# filter out COSMO-RS calcs
df = df[df['dtype'] != 'COSMO-RS']

# Labeling
XMIN : float = -0.1
XMAX : float = 1.1

ax.plot([XMIN, XMAX], [XMIN, XMAX], 'k--', lw=1)
ax.set_xlabel('Experimental $f_{zwit}$')
ax.set_ylabel('Predicted $f_{zwit}$')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(XMIN, XMAX)
# Create a legend mapping colors to source
handles, _ = scatter.legend_elements(prop="colors")
labels = pd.Categorical(df['source']).categories
ax.legend(handles, labels, title="Source", loc='upper left', bbox_to_anchor=(1,1))

# Layout and save
plt.tight_layout()
plt.savefig('f_zwit_scatter.png', dpi=300, bbox_inches='tight')

# Transform fz to Kz by plotting Kz = fz / (1 - fz)
# Plot exp't Kz vs predicted Kz
# Plot the log10 transform.
# if Kz is 0, drop it
KZ_exp = np.log10(df['experimental_f_zwit'] / (1 - df['experimental_f_zwit']))
KZ_pred = np.log10(df['predicted_f_zwit'] / (1 - df['predicted_f_zwit']))
valid = np.isfinite(KZ_exp) & np.isfinite(KZ_pred)
kz_exp_valid = KZ_exp[valid]
kz_pred_valid = KZ_pred[valid]

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.scatter(kz_exp_valid, kz_pred_valid, c='red', label='Kz')
ax.set_xlabel(r'Experimental $\log_{10}$ $K_{\mathrm{zwit}}$', fontsize=20)
ax.set_ylabel(r'Predicted $\log_{10}$ $K_{\mathrm{zwit}}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

xmin_k = np.min((np.min(kz_exp_valid - 1), np.min(kz_pred_valid - 1)))
xmax_k = np.max((np.max(kz_exp_valid + 1), np.max(kz_pred_valid + 1)))
ax.plot([xmin_k, xmax_k], [xmin_k, xmax_k], 'k--', lw=1)
ax.set_xlim(xmin_k, xmax_k)
ax.set_ylim(xmin_k, xmax_k)

# highlight y=x+/-1 line from parity line, y=x+/-2 line
ax.plot([xmin_k, xmax_k], [xmin_k + 1, xmax_k + 1], 'k--', lw=1, alpha=0.5)
ax.plot([xmin_k, xmax_k], [xmin_k - 1, xmax_k - 1], 'k--', lw=1, alpha=0.5)
ax.plot([xmin_k, xmax_k], [xmin_k + 2, xmax_k + 2], 'k--', lw=1, alpha=0.2)
ax.plot([xmin_k, xmax_k], [xmin_k - 2, xmax_k - 2], 'k--', lw=1, alpha=0.2)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

n = len(kz_exp_valid)
mae = np.mean(np.abs(kz_pred_valid - kz_exp_valid))
rmse = np.sqrt(np.mean((kz_pred_valid - kz_exp_valid) ** 2))
stats_text = f'N = {n}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'
ax.text(
    0.97, 0.03, stats_text,
    transform=ax.transAxes,
    fontsize=20,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
)

plt.tight_layout()
plt.savefig('kz_scatter_log10.svg', bbox_inches='tight', transparent=True)
plt.savefig('kz_scatter_log10.png', dpi=300, bbox_inches='tight', transparent=True)
