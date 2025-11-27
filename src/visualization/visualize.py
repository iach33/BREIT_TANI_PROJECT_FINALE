import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_grid_hist(df, cols=None, ncols=3, bins=20, title=None, save_path=None):
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()
        cols = [c for c in cols if not c.endswith('_n__rows')]

    n = len(cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for i, c in enumerate(cols):
        r, k = divmod(i, ncols)
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        axes[r, k].hist(s, bins=bins)
        axes[r, k].set_title(c)
        axes[r, k].grid(True, alpha=0.3)

    for j in range(n, nrows*ncols):
        r, k = divmod(j, ncols)
        axes[r, k].axis('off')

    if title: fig.suptitle(title, y=1.02)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Chart saved as: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_grid_hist_by_deficit(df, cols=None, tag='deficit', ncols=3, bins=20, title=None, density=False, alpha=0.6, save_path=None):
    if tag not in df.columns:
        raise ValueError(f"La columna de tag '{tag}' no existe en el DataFrame")

    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()
    
    cols = [c for c in cols if c not in {tag} and not c.endswith('_n__rows')]

    n = len(cols)
    if n == 0:
        raise ValueError("No hay columnas numéricas para graficar")

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    m0 = df[tag] == 0
    m1 = df[tag] == 1

    for i, c in enumerate(cols):
        r, k = divmod(i, ncols)
        ax = axes[r, k]

        s0 = pd.to_numeric(df.loc[m0, c], errors='coerce').dropna()
        s1 = pd.to_numeric(df.loc[m1, c], errors='coerce').dropna()

        if len(s0) == 0 and len(s1) == 0:
            ax.set_title(c)
            ax.text(0.5, 0.5, "sin datos", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        if isinstance(bins, int):
            data_all = np.concatenate([s0.values, s1.values]) if len(s0) and len(s1) else (s0.values if len(s0) else s1.values)
            bin_edges = np.histogram_bin_edges(data_all, bins=bins)
        else:
            bin_edges = bins

        if len(s0):
            ax.hist(s0, bins=bin_edges, alpha=alpha, label='deficit=0', color='blue', density=density)
        if len(s1):
            ax.hist(s1, bins=bin_edges, alpha=alpha, label='deficit=1', color='red', density=density)

        ax.set_title(c)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    for j in range(n, nrows*ncols):
        r, k = divmod(j, ncols)
        axes[r, k].axis('off')

    if title: fig.suptitle(title, y=1.02)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Chart saved as: {save_path}")
    else:
        plt.show()
    plt.close()
