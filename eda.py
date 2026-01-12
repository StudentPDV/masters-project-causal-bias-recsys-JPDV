import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_PATH = ('processed_movielens_m32.pkl')
sns.set_style("whitegrid") 
plt.rcParams.update({'font.size': 12}) 


def main():
    print("--- GENERATING DATASET ANALYSIS FIGURES ---")

    print("1. Loading Data (this may take a moment)...")
    df = pd.read_pickle(DATA_PATH)
    
    n_users = df['userId'].nunique()
    n_items = df['movieId'].nunique()
    n_ratings = len(df)
    sparsity = 1 - (n_ratings / (n_users * n_items))

    print("\n=== DATASET STATISTICS ===")
    print(f"Total Ratings: {n_ratings:,}")
    print(f"Total Users:   {n_users:,}")
    print(f"Total Movies:  {n_items:,}")
    print(f"Sparsity:      {sparsity:.4%}")
    print("==========================\n")

    print("2. Generating User Mean Histogram...")

    user_means = df.groupby('userId')['rating'].mean()

    plt.figure(figsize=(10, 6))
    sns.histplot(user_means, bins=50, kde=True, color='#2c3e50', edgecolor='black', alpha=0.7)

    global_mean = df['rating'].mean()
    plt.axvline(global_mean, color='#e74c3c', linestyle='--', linewidth=2, label=f'Global Mean ({global_mean:.2f})')

    plt.title('Distribution of User Average Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Average Rating given by User', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig1_User_Rating_Distribution.png', dpi=300)
    print("   -> Saved 'Fig1_User_Rating_Distribution.png'")

    print("3. Generating Long Tail Plot...")

    item_counts = df['movieId'].value_counts().values

    plt.figure(figsize=(10, 6))
    plt.plot(item_counts, color='#2980b9', linewidth=2)

    top_20_idx = int(len(item_counts) * 0.2)
    plt.fill_between(range(top_20_idx), item_counts[:top_20_idx], color='#2980b9', alpha=0.3, label='Top 20% (Popular)')

    bottom_20_idx = int(len(item_counts) * 0.8)
    plt.fill_between(range(bottom_20_idx, len(item_counts)), item_counts[bottom_20_idx:], color='#27ae60', alpha=0.3,
                     label='Bottom 20% (Niche)')

    plt.title('The Long Tail: Rating Frequency by Movie', fontsize=14, fontweight='bold')
    plt.xlabel('Movie Rank (Sorted by Popularity)', fontsize=12)
    plt.ylabel('Number of Ratings (Log Scale)', fontsize=12)
    plt.yscale('log')  # Log scale is crucial for Long Tail plots
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig2_Long_Tail_Distribution.png', dpi=300)
    print("   -> Saved 'Fig2_Long_Tail_Distribution.png'")

    print("4. Generating Recency Histogram...")

    ages = df[df['item_age'] >= 0]['item_age']

    plt.figure(figsize=(10, 6))
    sns.histplot(ages, bins=40, color='#8e44ad', edgecolor='black', alpha=0.7)

    plt.axvline(1, color='#f1c40f', linestyle='--', linewidth=2, label='Recency Cutoff (1 Year)')

    plt.title('Distribution of Movie Age at Time of Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Age of Movie (Years)', fontsize=12)
    plt.ylabel('Number of Ratings', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig3_Item_Age_Distribution.png', dpi=300)
    print("   -> Saved 'Fig3_Item_Age_Distribution.png'")

    print("\n--- DONE! All figures saved. ---")

if __name__ == "__main__":
    main()
