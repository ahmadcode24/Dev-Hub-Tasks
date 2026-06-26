"""
visualizations.py
-----------------
Publication-quality plotting functions for Iris EDA.
All functions save figures to outputs/figures/ automatically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

# Global style configuration
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

PALETTE = {'setosa': '#2E86AB', 'versicolor': '#A23B72', 'virginica': '#F18F01'}
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_univariate_distributions(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Histograms with KDE overlays for all features by species."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        for species in df['species'].unique():
            subset = df[df['species'] == species][feature]
            sns.histplot(subset, kde=True, bins=15, alpha=0.4,
                         label=species, ax=ax, stat='density', linewidth=1.5)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(title='Species', loc='best', frameon=True)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Univariate Feature Distributions by Species', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '01_univariate_distributions.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_boxplots(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Box plots for outlier detection and spread analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        sns.boxplot(data=df, x='species', y=feature, palette=PALETTE, 
                    ax=ax, linewidth=1.5)
        ax.set_title(f'{feature.replace("_", " ").title()} by Species', 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Species', fontsize=11)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Box Plots: Feature Distributions & Outlier Detection', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '02_boxplots.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Correlation matrix heatmap with upper triangle masked."""
    corr_matrix = df.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                cmap='RdBu_r', center=0, square=True, linewidths=1,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'}, ax=ax)

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '04_correlation_heatmap.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_petal_scatter(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Focused scatter plot of the two most discriminative features."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for species in df['species'].unique():
        subset = df[df['species'] == species]
        ax.scatter(subset['petal_length'], subset['petal_width'],
                   label=species.title(), alpha=0.75, s=100,
                   edgecolors='white', linewidth=1, color=PALETTE[species])

    ax.set_xlabel('Petal Length (cm)', fontsize=12)
    ax.set_ylabel('Petal Width (cm)', fontsize=12)
    ax.set_title('Petal Dimensions: Primary Classification Features', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Species', fontsize=11, title_fontsize=12, 
              frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    ax.annotate('Setosa is perfectly linearly\nseparable using petal features.',
                xy=(1.5, 0.25), xytext=(3.5, 1.0),
                fontsize=10, color='#2E86AB', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8', 
                         edgecolor='#2E86AB'))

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '05_petal_scatter.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig
