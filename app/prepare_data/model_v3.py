import duckdb
import os
import numpy as np
import gensim
import pickle

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV



class MovieRatingPredictor:
    def __init__(self, df, word2vec_params=None, model_type=LinearRegression, n_jobs=1):
        """
        Initializes the MovieRatingPredictor with the dataframe and optional word2vec parameters.

        Args:
            df: The dataframe containing 'overview', 'genre_ids', and 'vote_average' columns.
            word2vec_params: A dictionary of parameters for the word2vec model (optional).
        """
        self.df = df
        self.word2vec_params = word2vec_params or {'vector_size': 100, 'window': 5, 'min_count': 1, 'sg': 0}
        self.word2vec_model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.mlb = None  # Store the MultiLabelBinarizer

        print(f"Model Type: {model_type}")

        if model_type == 'decision_tree':
            self.model = DecisionTreeRegressor()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor()
        elif model_type == 'svr':
            self.model = SVR()
        elif model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), 
                activation='relu', 
                solver='adam', 
                batch_size=512, 
                early_stopping=True
            )
        else:  # Default to Linear Regression
            self.model = LinearRegression(n_jobs=n_jobs)

    def preprocess_text(self, text):
        """
        Preprocesses the text data (implementation depends on your specific needs).

        Args:
            text: The text to be preprocessed.

        Returns:
            The preprocessed text.
        """
        return text.lower().split()

    def train_word2vec(self, field_name):
        """
        Trains the word2vec model on the preprocessed 'overview' data.
        """
        with mlflow.start_run(nested=True, run_name="word2vec_training"):
            print("Training Word2Vec")
            self.df[f'{field_name}_processed'] = self.df[field_name].apply(self.preprocess_text)
            sentences = self.df[f'{field_name}_processed'].tolist()
            self.word2vec_model = gensim.models.Word2Vec(sentences, **self.word2vec_params)

            # Log Word2Vec parameters
            mlflow.log_params(self.word2vec_params)

    def get_embedding(self, text):
        """
        Generates the word2vec embedding for the given text.

        Args:
            text: The text to be embedded.

        Returns:
            The word2vec embedding (mean of word vectors) or a zero vector if no words are found.
        """

        word_vectors = [self.word2vec_model.wv[w] for w in text if w in self.word2vec_model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.word2vec_params['vector_size'])

    def prepare_features(self, text_field_name, classification_field):
        """
        Prepares the features (word2vec embeddings and encoded 'genre_ids') for the model.
        """

        print("Prepare features")
        self.df[f'{text_field_name}_embedding'] = self.df[f'{text_field_name}_processed'].apply(lambda x: self.get_embedding(x))

        # Multi-hot encoding for 'genre_ids'
        self.mlb = MultiLabelBinarizer()  # Initialize and store the MultiLabelBinarizer
        genre_ids_encoded = self.mlb.fit_transform(self.df['genre_ids'])

        X = np.hstack((self.df[f'{text_field_name}_embedding'].tolist(), genre_ids_encoded))
        y = self.df[classification_field]
        return X, y

    def train_and_evaluate(self, text_field, classification_field):
        """
        Trains the regression model and evaluates its performance.
        """
        with mlflow.start_run(nested=True, run_name=f"train_and_evaluate_{self.model.__class__.__name__}"):
            print("Train and evaluate")
            X, y = self.prepare_features(text_field, classification_field)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)


            mse = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')


            # Log model parameters and metrics
            mlflow.log_params(self.model.get_params())
            mlflow.log_metric("mse", mse)

            # Save the trained model
            mlflow.sklearn.log_model(self.model, "model")


    def save_model(self, filename='movie_rating_model.pkl'):
        """
        Saves the trained model to a pickle file.

        Args:
            filename: The name of the pickle file to save the model to (default: 'movie_rating_model.pkl').
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename='movie_rating_model.pkl'):
        """
        Loads a trained model from a pickle file.

        Args:
            filename: The name of the pickle file to load the model from (default: 'movie_rating_model.pkl').

        Returns:
            The loaded MovieRatingPredictor instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def predict(self, text, genre_ids):
        """
        Predicts the movie rating based on the provided text and genre IDs.

        Args:
            text: The overview text of the movie.
            genre_ids: A list of genre IDs for the movie.

        Returns:
            The predicted movie rating.
        """
        processed_text = self.preprocess_text(text)
        text_embedding = self.get_embedding(processed_text)

        # Ensure consistent genre encoding
        genre_ids_encoded = self.mlb.transform([genre_ids]) 

        text_embedding = text_embedding.reshape(1, -1) 

        X = np.hstack((text_embedding, genre_ids_encoded))
        return self.model.predict(X)[0]



tmdb_token = os.getenv("TMDB_TOKEN")
output_movies_folder = os.getenv("OUTPUT_MOVIES_FOLDER")
mlflow_track_server = os.getenv("MLFLOW_TRACK_SERVER")

mlflow.set_tracking_uri(mlflow_track_server)
#output_movies_folder="C:/projects/fiap-proj-int-03/app/get_movies/output_files/*.json"

conn = duckdb.connect(config = {'threads': 5})

conn.execute(f"CREATE TABLE movies AS (SELECT * FROM read_json('{output_movies_folder}'));")

#df = conn.execute(f"SELECT genre_ids, title, vote_average FROM read_json('{output_movies_folder}/*.json')").fetchdf()

df = conn.execute(f"""
    SELECT 
        --*
        id
        , genre_ids
        , title
        , vote_average
        , overview
    FROM movies
    WHERE overview is not null 
      AND trim(overview) <> ''
                  
""").fetchdf()

print(f"number of lines: {df.count()}")

def train():
    model_type="linear_regression"
    with mlflow.start_run(run_name="train_{model_type}"):
        # Example Usage:
        predictor = MovieRatingPredictor(
            df,
            word2vec_params={
                "vector_size": 200,
                "window": 10,
                "min_count": 2,
                "sg": 0,
                "workers": 8,
            },
            model_type=model_type,  # Make sure this matches the model you trained
            n_jobs=10,
        )
        predictor.train_word2vec("overview")
        predictor.train_and_evaluate("overview", "vote_average")

def train_random_forest():
    with mlflow.start_run(run_name="random_forest_tuning"):
        # Example Usage:
        predictor = MovieRatingPredictor(
            df,
            word2vec_params={
                "vector_size": 200,
                "window": 10,
                "min_count": 2,
                "sg": 0,
                "workers": 8,
            },
            model_type="random_forest",
            n_jobs=10,
        )
        predictor.train_word2vec("overview")

        # Log Word2Vec parameters
        mlflow.log_params(predictor.word2vec_params)

        # Hyperparameter Tuning for RandomForestRegressor
        param_grid = {
            "n_estimators": [500, 1000, 2000],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }

        grid_search = GridSearchCV(
            predictor.model, param_grid, cv=5, scoring="neg_mean_squared_error"
        )
        X, y = predictor.prepare_features("overview", "vote_average")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        grid_search.fit(X_train, y_train)

        # Update the model with the best parameters
        predictor.model = grid_search.best_estimator_

        # Evaluate the tuned model
        y_pred = predictor.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error after tuning: {mse}")

        # Log best parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mse", mse)

        # Save the trained model (IMPORTANT: Do this AFTER training & tuning)
        mlflow.sklearn.log_model(predictor.model, "model") 
        # predictor.save_model() # You can keep this if you also want to save locally


def test():
    # Later, you can load and use the model
    # loaded_predictor = MovieRatingPredictor.load_model() # If you saved locally
    with mlflow.start_run(run_name="test_predictions"):
        # Load the model from MLflow
        logged_model = 'runs:/fiap-03/model'  # Replace <RUN_ID> with the actual run ID from MLflow UI
        loaded_predictor = mlflow.sklearn.load_model(logged_model)

        print("Model loaded")



        new_movie_overview = "A small team of scientists must race against time to stop what seems to be a cascade of global disasters signaling the possible apocalypse and end of days."
        new_movie_genre_ids = [878, 27] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (60): {predicted_rating}. Distance: {predicted_rating - 6.00}")


        new_movie_overview = "Heather bumps into Carla, having not spoken to her in years, and presents her with a very unexpected proposition that could change both of their lives forever."
        new_movie_genre_ids = [878, 27] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (58): {predicted_rating}. Distance: {predicted_rating - 5.80}")

        new_movie_overview = "In a futuristic dystopia with enforced beauty standards, a teen awaiting mandatory cosmetic surgery embarks on a journey to find her missing friend."
        new_movie_genre_ids = [878, 12] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (60): {predicted_rating}. Distance: {predicted_rating - 6.00}")


        new_movie_overview = "A talented martial artist who can't walk past a person in need unites with a probation officer to fight and prevent crime as a martial arts officer."
        new_movie_genre_ids = [28, 35, 80] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (79): {predicted_rating}. Distance: {predicted_rating - 7.90}")

        new_movie_overview = "A talented martial artist who can't walk past a person in need unites with a probation officer to fight and prevent crime as a martial arts officer."
        new_movie_genre_ids = [28, 35, 80] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (79): {predicted_rating}. Distance: {predicted_rating - 7.90}")

        new_movie_overview = "A detective begins to investigate a series of mysterious murders that are connected to a demonic book that brings dolls to life. As the body count begins to rise, the detective soon learns the curse of the demonic Friday and must find a way to stop it before any others disappear."
        new_movie_genre_ids = [27] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (32): {predicted_rating}. Distance: {predicted_rating - 3.20}")


        new_movie_overview = "When a group of ex-military members is hired to retrieve a lost bag of stolen money, their mission becomes more difficult after a lone hunter finds the bag first."
        new_movie_genre_ids = [28, 53, 10770] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (45): {predicted_rating}. Distance: {predicted_rating - 4.50}")

        new_movie_overview = "A young teenager named Mikey Walsh finds an old treasure map in his father's attic. Hoping to save their homes from demolition, Mikey and his friends Data Wang, Chunk Cohen, and Mouth Devereaux run off on a big quest to find the secret stash of Pirate One-Eyed Willie."
        new_movie_genre_ids = [12, 35, 10751] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (75): {predicted_rating}. Distance: {predicted_rating - 7.50}")


        new_movie_overview = "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope."
        new_movie_genre_ids = [18, 80] 
        predicted_rating = loaded_predictor.predict(new_movie_overview, new_movie_genre_ids)
        print(f"Predicted rating for the new movie (87): {predicted_rating}. Distance: {predicted_rating - 8.70}")

train()
#train_random_forest()
test()