import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv('results/f_zwit_benchmark_with_empir_cxns/benchmark_results.csv')

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
ax.set_title('Experimental vs Predicted $f_{zwit}$')

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
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(KZ_exp[KZ_exp != -np.inf], KZ_pred[KZ_exp != -np.inf], c='red', label='Kz')
ax.set_xlabel('Experimental log10($K_{zwit}$)')
ax.set_ylabel('Predicted log10($K_{zwit}$)')

xmin_k = np.min((np.min(KZ_exp[KZ_exp != -np.inf] - 1), np.min(KZ_pred - 1)))
xmax_k = np.max((np.max(KZ_exp[KZ_exp != np.inf] + 1), np.max(KZ_pred + 1)))
ax.plot([xmin_k, xmax_k], [xmin_k, xmax_k], 'k--', lw=1)
ax.set_xlim(xmin_k, xmax_k)
ax.set_ylim(xmin_k, xmax_k)
ax.set_title('Experimental vs Predicted log10($K_{zwit}$)')

# highlight y=x+/-1 line from parity line, y=x+/-2 line
ax.plot([xmin_k, xmax_k], [xmin_k+1, xmax_k+1], 'k--', lw=1, alpha=0.5)
ax.plot([xmin_k, xmax_k], [xmin_k-1, xmax_k-1], 'k--', lw=1, alpha=0.5)
ax.plot([xmin_k, xmax_k], [xmin_k+2, xmax_k+2], 'k--', lw=1, alpha=0.2)
ax.plot([xmin_k, xmax_k], [xmin_k-2, xmax_k-2], 'k--', lw=1, alpha=0.2)

#  save
plt.tight_layout()
plt.savefig('kz_scatter_log10.png', dpi=300, bbox_inches='tight')
