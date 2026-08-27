import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from pathlib import Path
from datetime import datetime

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


class CollaborativeFilteringTrainer:

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.model = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}

    def load_data(self):
        print("Loading Data")

        interactions_file = self.data_path / "user_interactions.csv"
        if not interactions_file.exists():
            raise FileNotFoundError("user_interactions.csv not found. Run parse_user_interactions.py first.")

        print(f"\nLoading interactions from {interactions_file}...")
        interactions_df = pd.read_csv(interactions_file)
        print(f"Loaded {len(interactions_df):,} interactions")

        tracks_file = self.data_path / "tracks.csv"
        if not tracks_file.exists():
            raise FileNotFoundError("tracks.csv not found. Run parse_lastfm_tracks.py first.")

        print(f"Loading track metadata from {tracks_file}...")
        tracks_df = pd.read_csv(tracks_file, low_memory=False)
        print(f"Loaded {len(tracks_df):,} tracks")

        return interactions_df, tracks_df

    def create_interaction_matrix(self, interactions_df: pd.DataFrame):
        print("Creating Interaction Matrix")

        unique_users = interactions_df['user_id'].unique()
        unique_tracks = interactions_df['track_id'].unique()

        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {track: idx for idx, track in enumerate(unique_tracks)}
        self.reverse_item_mapping = {idx: track for track, idx in self.item_mapping.items()}

        print(f"\nMatrix dimensions:")
        print(f"  Users  : {len(unique_users):,}")
        print(f"  Tracks : {len(unique_tracks):,}")

        user_indices = interactions_df['user_id'].map(self.user_mapping).values
        track_indices = interactions_df['track_id'].map(self.item_mapping).values
        play_counts = interactions_df['play_count'].values.astype(np.float32)

        #shape: (users x tracks) - this is what modern implicit expects
        self.user_item_matrix = csr_matrix(
            (play_counts, (user_indices, track_indices)),
            shape=(len(unique_users), len(unique_tracks))
        )
        #shape: (tracks x users) - kept for get_popular_tracks()
        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        total = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        filled = self.user_item_matrix.nnz
        print(f"\nMatrix statistics:")
        print(f"  user_item_matrix shape : {self.user_item_matrix.shape}  <- rows=users, cols=tracks")
        print(f"  item_user_matrix shape : {self.item_user_matrix.shape}  <- rows=tracks, cols=users")
        print(f"  Sparsity               : {1 - filled/total:.4%}")
        print(f"  Avg tracks per user    : {filled / len(unique_users):.1f}")
        print(f"  Avg listeners per track: {filled / len(unique_tracks):.1f}")

    def train_model(self, factors: int = 50, regularization: float = 0.01,
                    iterations: int = 15, alpha: float = 40.0):
        print("Training ALS Model")

        print(f"\nModel parameters:")
        print(f"  Factors        : {factors}")
        print(f"  Regularization : {regularization}")
        print(f"  Iterations     : {iterations}")
        print(f"  Alpha          : {alpha}")

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
            use_gpu=False,
        )

        print(f"\nTraining...")
        print(f"  Passing user_item_matrix (shape {self.user_item_matrix.shape}) to fit()")
        print(f"  Alpha applied by multiplying matrix: user_item_matrix * {alpha}")

        #apply alpha by multiplying the matrix, not via constructor param
        # (implicit >=0.6 removed alpha from constructor)
        self.model.fit(self.user_item_matrix * alpha, show_progress=True)

        #sanity check
        print(f"\nSanity check:")
        print(f"  model.item_factors shape : {self.model.item_factors.shape}")
        print(f"  model.user_factors shape : {self.model.user_factors.shape}")
        print(f"  item_mapping size        : {len(self.item_mapping):,}")
        print(f"  user_mapping size        : {len(self.user_mapping):,}")

        item_ok = self.model.item_factors.shape[0] == len(self.item_mapping)
        user_ok = self.model.user_factors.shape[0] == len(self.user_mapping)

        if not item_ok or not user_ok:
            raise RuntimeError(
                f"Shape mismatch after training!\n"
                f"  item_factors: {self.model.item_factors.shape[0]} vs item_mapping: {len(self.item_mapping)}\n"
                f"  user_factors: {self.model.user_factors.shape[0]} vs user_mapping: {len(self.user_mapping)}\n"
                f"Matrix orientation is still wrong. Check implicit library version."
            )

        print(f"  PASS: shapes match item/user mappings correctly")
        print("\nTraining complete!")

    def save_model(self, output_dir: str = "models"):
        print("Saving Model")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_file = output_path / f"als_model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved    : {model_file}")

        mappings_file = output_path / f"mappings_{timestamp}.pkl"
        with open(mappings_file, 'wb') as f:
            pickle.dump({
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
            }, f)
        print(f"Mappings saved : {mappings_file}")

        matrices_file = output_path / f"matrices_{timestamp}.pkl"
        with open(matrices_file, 'wb') as f:
            pickle.dump({
                'user_item_matrix': self.user_item_matrix,
                'item_user_matrix': self.item_user_matrix,
            }, f)
        print(f"Matrices saved : {matrices_file}")

        metadata_file = output_path / "latest_model_info.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Latest model timestamp: {timestamp}\n")
            f.write(f"Model file: als_model_{timestamp}.pkl\n")
            f.write(f"Mappings file: mappings_{timestamp}.pkl\n")
            f.write(f"Matrices file: matrices_{timestamp}.pkl\n")
            f.write(f"Users: {len(self.user_mapping):,}\n")
            f.write(f"Tracks: {len(self.item_mapping):,}\n")
        print(f"Metadata saved : {metadata_file}")

        return timestamp


def main():
    print("Collaborative Filtering Model Training")

    trainer = CollaborativeFilteringTrainer(data_path="data")
    interactions_df, tracks_df = trainer.load_data()

    use_sample = input("\nUse sample data for testing? (y/n, default=n): ").lower()
    if use_sample == 'y':
        sample_size = 100_000
        print(f"\nUsing sample of {sample_size:,} interactions")
        interactions_df = interactions_df.sample(
            n=min(sample_size, len(interactions_df)), random_state=42
        )

    trainer.create_interaction_matrix(interactions_df)
    trainer.train_model(factors=50, regularization=0.01, iterations=15, alpha=40.0)
    timestamp = trainer.save_model(output_dir="models")

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Model timestamp: {timestamp}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()