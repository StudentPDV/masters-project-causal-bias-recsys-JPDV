import pandas as pd
import numpy as np


def prepare_split_data(ratings_path, movies_path):
    print(f"--- STARTING DATA PREPARATION (SPLIT MODE) ---")

    print(f"1. Loading and calculating features...")
    df = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    df['rating_year'] = pd.to_datetime(df['timestamp'], unit='s').dt.year
    df = df.merge(movies[['movieId', 'release_year']], on='movieId', how='left')
    df['item_age'] = df['rating_year'] - df['release_year']
    df = df[df['item_age'] >= 0]

    item_stats = df.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()

    user_stats = df.groupby('userId')['rating'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['userId', 'user_mean', 'user_std']

    df = df.merge(item_stats, on='movieId', how='left')
    df = df.merge(user_stats, on='userId', how='left')

    df['user_std'] = df['user_std'].replace(0, 1)
    df['rating_norm'] = (df['rating'] - df['user_mean']) / df['user_std']

    pop_threshold = df['num_ratings'].quantile(0.8)
    nov_threshold = df['num_ratings'].quantile(0.2)
    global_mean = df['rating'].mean()

    df['T_pop'] = (df['num_ratings'] >= pop_threshold).astype(int)
    df['T_nov'] = (df['num_ratings'] <= nov_threshold).astype(int)
    df['T_rec'] = (df['item_age'] <= 1).astype(int)
    df['T_scale'] = (df['user_mean'] >= global_mean).astype(int)

    df['Y_raw'] = (df['rating'] >= 4.0).astype(int)
    df['Y_norm'] = (df['rating_norm'] > 0).astype(int)

    print("\n2. Splitting and Saving Optimized Files...")

    cols_pop = ['T_pop', 'Y_raw', 'Y_norm', 'avg_rating']
    df[cols_pop].to_pickle('data_popularity.pkl')
    print(f"   -> Saved 'data_popularity.pkl' (Columns: {cols_pop})")

    cols_nov = ['T_nov', 'Y_raw', 'Y_norm', 'avg_rating']
    df[cols_nov].to_pickle('data_novelty.pkl')
    print(f"   -> Saved 'data_novelty.pkl' (Columns: {cols_nov})")

    cols_rec = ['T_rec', 'Y_raw', 'Y_norm', 'avg_rating']
    df[cols_rec].to_pickle('data_recency.pkl')
    print(f"   -> Saved 'data_recency.pkl' (Columns: {cols_rec})")

    cols_scale = ['T_scale', 'Y_raw', 'Y_norm', 'avg_rating']
    df[cols_scale].to_pickle('data_userscale.pkl')
    print(f"   -> Saved 'data_userscale.pkl' (Columns: {cols_scale})")

    print("\n--- SUCCESS! 4 files created. ---")

if __name__ == "__main__":
    dataset = 'm32'
    prepare_split_data(f"data/{dataset}/ratings.csv", f'data/{dataset}/movies.csv')
