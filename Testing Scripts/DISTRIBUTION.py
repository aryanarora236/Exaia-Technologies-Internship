import pandas as pd
import matplotlib.pyplot as plt

# Load distributions
manual = pd.read_csv("manual_distribution.csv")
automated = pd.read_csv("automated_distribution.csv")

for trait in ["age", "gender", "education"]:
    # Filter for this trait
    df_m = manual[manual["Trait"] == trait]
    df_a = automated[automated["Trait"] == trait]
    
    # Determine all bins
    bins = sorted(set(df_m["Normalized"]).union(df_a["Normalized"]))
    
    # Counts for each bin
    counts_m = [df_m[df_m["Normalized"] == b]["Count"].sum() for b in bins]
    counts_a = [df_a[df_a["Normalized"] == b]["Count"].sum() for b in bins]
    
    # X locations
    x = range(len(bins))
    width = 0.35
    
    # Plot
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], counts_m, width, label="Manual")
    ax.bar([i + width/2 for i in x], counts_a, width, label="Automated")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {trait.capitalize()}")
    ax.legend()
    plt.tight_layout()
    plt.show()
