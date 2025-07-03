import pandas as pd
import matplotlib.pyplot as plt

# Cargar los archivos CSV
df_p = pd.read_csv('cluster_assignments_comparison_P.csv')  # Sustituye con la ruta real del primer CSV
df_s = pd.read_csv('cluster_assignments_comparison_S.csv')  # Sustituye con la ruta real del segundo CSV


# Extract numeric peptide ID (as integer)
df_p['Base'] = df_p['Peptide'].str.replace('_P', '', regex=False).astype(int)
df_s['Base'] = df_s['Peptide'].str.replace('_S', '', regex=False).astype(int)

# Merge on numeric base
merged = pd.merge(df_p, df_s, on='Base', suffixes=('_P', '_S'))
merged = merged.sort_values('Base')  # Sort by peptide number

# Store match percentages
results = {}

for method in ['KMeans', 'Agglomerative', 'QuantumKMeans']:
    match_col = f'{method}_match'
    merged[match_col] = merged[f'{method}_P'] == merged[f'{method}_S']
    matches = merged[match_col].sum()
    total = len(merged)
    percentage = 100 * matches / total
    results[method] = percentage

    # Plotting
    colors = ['green' if m else 'red' for m in merged[match_col]]
    x_labels = merged['Base'].astype(str)

    plt.figure(figsize=(12, 5))
    bars = plt.bar(x_labels, merged[match_col].astype(int), color=colors)
    plt.title(f'{method} Cluster Match: {percentage:.1f}%', fontsize=14)
    plt.ylabel('Match (1 = Yes, 0 = No)', fontsize=12)
    plt.xlabel('Peptide ID', fontsize=12)
    plt.ylim(0, 1.2)
    plt.xticks(rotation=90)
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add ✓ and ✗ above bars
    for bar, match in zip(bars, merged[match_col]):
        symbol = '✓' if match else '✗'
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 symbol, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    # Save the figure
    filename = f'{method}_match.png'
    plt.savefig(filename, dpi=300)
    plt.close()

# Print summary
print("\n=== Match Percentage Summary ===")
for method, pct in results.items():
    print(f"{method}: {pct:.1f}% match")
