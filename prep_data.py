import pandas as pd
import numpy as np


def prepare_movielens_data(ratings_path, movies_path, output_path):
    print(f"--- STARTING DATA PREPARATION ---")

    print(f"1. Loading data from {ratings_path}...")
    try:
        df = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
    except FileNotFoundError:
        print("Error: Files not found. Check paths.")
        return

    print("2. Extracting timestamps and release years...")
    movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    df['rating_year'] = pd.to_datetime(df['timestamp'], unit='s').dt.year

    df = df.merge(movies[['movieId', 'release_year']], on='movieId', how='left')

    df['item_age'] = df['rating_year'] - df['release_year']
    df = df[df['item_age'] >= 0]  # Filter invalid dates

    print("3. Calculating Statistics...")
    item_stats = df.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()

    user_stats = df.groupby('userId')['rating'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['userId', 'user_mean', 'user_std']

    df = df.merge(item_stats, on='movieId', how='left')
    df = df.merge(user_stats, on='userId', how='left')

    print("4. Normalizing...")
    df['user_std'] = df['user_std'].replace(0, 1)
    df['rating_norm'] = (df['rating'] - df['user_mean']) / df['user_std']

    print("5. Defining Variables...")

    pop_threshold = df['num_ratings'].quantile(0.8)
    nov_threshold = df['num_ratings'].quantile(0.2)
    global_mean = df['rating'].mean()

    df['T_pop'] = (df['num_ratings'] >= pop_threshold).astype(int)
    df['T_nov'] = (df['num_ratings'] <= nov_threshold).astype(int)
    df['T_rec'] = (df['item_age'] <= 1).astype(int)  # Rated within 1 year
    df['T_scale'] = (df['user_mean'] >= global_mean).astype(int)

    df['Y_raw'] = (df['rating'] >= 4.0).astype(int)
    df['Y_norm'] = (df['rating_norm'] > 0).astype(int)

    print(f"6. Saving to {output_path}...")

    df.to_pickle(output_path)

    print(f"--- SUCCESS! Saved to {output_path} ---")


if __name__ == "__main__":
    dataset = 'm32'
    prepare_movielens_data(f"data/{dataset}/ratings.csv", f'data/{dataset}/movies.csv', f'processed_movielens_{dataset}.pkl')
