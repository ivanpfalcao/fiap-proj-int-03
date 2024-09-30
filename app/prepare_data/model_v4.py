import duckdb
import os
import logging
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import torch
import xgboost as xgb


from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MovieRatingPredictor:
	def __init__(self, model_type_s="LinearRegression", n_jobs=1):
		"""
		Initializes the MovieRatingPredictor with the dataframe.
		"""
		mlflow_track_server = os.getenv("MLFLOW_TRACK_SERVER")

		mlflow.set_tracking_uri(mlflow_track_server)

		self.encoder = OneHotEncoder(handle_unknown='ignore')
		self.mlb = None  # Store the MultiLabelBinarizer
		self.model_type = model_type_s
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	
		# Load a more advanced BERT model for richer embeddings
		self.bert_model = SentenceTransformer('all-MPNet-base-v2')  # Richer BERT model with 768 dimensions

		logging.info(f"Model Type: {self.model_type}")

		# Initialize the model based on the user's choice
		if self.model_type == 'decision_tree':
			self.model = DecisionTreeRegressor(max_depth=10)
		elif self.model_type == 'random_forest':
			self.model = RandomForestRegressor(
				max_features=0.4,
				n_estimators=1000,
				max_depth=50,
				min_samples_split=10,
			)
		elif self.model_type == 'svr':
			self.model = SVR(kernel='sigmoid', C=1.0, epsilon=0.1)
		elif self.model_type == 'mlp':
			# The basic MLP setup (will be tuned later)
			self.model = MLPRegressor(
				hidden_layer_sizes=(256, 128, 64),
				activation='relu',
				solver='adam',
				alpha=0.001,
				learning_rate_init=0.001,
				early_stopping=True,
				max_iter=1000
			)
		elif self.model_type == 'xgboost':
			self.model = xgb.XGBRegressor(
				objective='reg:squarederror',
				n_estimators=100,
				max_depth=6,
				learning_rate=0.1,
				n_jobs=n_jobs
			)
		else:  # Default to Linear Regression
			self.model = LinearRegression(n_jobs=n_jobs)

	def tune_mlp_hyperparameters(self, X_train, y_train):
		"""
		Perform hyperparameter tuning for the MLP model using GridSearchCV.
		"""
		param_grid = {
			'hidden_layer_sizes': [(512, 256, 128), (256, 128, 64), (128, 64, 32)],
			'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
			'learning_rate_init': [0.001, 0.01],
			'solver': ['adam', 'sgd'],
			'early_stopping': [True]
		}

		grid_search = GridSearchCV(MLPRegressor(max_iter=1000), param_grid, cv=3)
		grid_search.fit(X_train, y_train)

		# Update the model with the best parameters
		self.model = grid_search.best_estimator_

		logging.info("Best hyperparameters found: ", grid_search.best_params_)

	def preprocess_text(self, text):
		"""
		Uses BERT to generate embeddings from the text data.
		"""
		return self.bert_model.encode(text, convert_to_tensor=False)

	def prepare_features(self, text_field_name, classification_field):
		"""
		Prepares the features (BERT embeddings and encoded 'genre_ids') for the model.
		"""
		logging.info("Prepare features")
		# Generate BERT embeddings for the text field
		self.input_df[f'{text_field_name}_embedding'] = self.input_df[text_field_name].apply(lambda x: self.preprocess_text(x))

		# Multi-hot encoding for 'genre_ids'
		self.mlb = MultiLabelBinarizer()
		genre_ids_encoded = self.mlb.fit_transform(self.input_df['genre_ids'])

		# Combine embeddings and encoded genre ids
		X = np.hstack((self.input_df[f'{text_field_name}_embedding'].tolist(), genre_ids_encoded))
		y = self.input_df[classification_field]
		return X, y

	def train_and_evaluate(self, text_field, classification_field):
		"""
		Trains the regression model and evaluates its performance.
		"""
		with mlflow.start_run(nested=True, run_name=f"train_and_evaluate_{self.model.__class__.__name__}"):
			logging.info("Train and evaluate")
			X, y = self.prepare_features(text_field, classification_field)
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

			# Normalize the input features (Scaling)
			scaler = StandardScaler()
			X_train_scaled = scaler.fit_transform(X_train)
			X_test_scaled = scaler.transform(X_test)

			# Hyperparameter tuning for MLP
			if self.model_type == 'mlp':
				self.tune_mlp_hyperparameters(X_train_scaled, y_train)

			# Train the model
			self.model.fit(X_train_scaled, y_train)

			# Make predictions
			y_pred = self.model.predict(X_test_scaled)

			# Evaluate the model
			mse = mean_squared_error(y_test, y_pred)
			r2 = r2_score(y_test, y_pred)
			mae = mean_absolute_error(y_test, y_pred)
			mape = mean_absolute_percentage_error(y_test, y_pred)

			logging.info(f'Mean Squared Error: {mse}')
			logging.info(f'R-squared: {r2}')
			logging.info(f'Mean Absolute Error: {mae}')
			logging.info(f'Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%')

			# Log model parameters, metrics, and timestamp
			mlflow.log_params(self.model.get_params())
			mlflow.log_metric("mse", mse)
			mlflow.log_metric("r2_score", r2)
			mlflow.log_metric("mae", mae)
			mlflow.log_metric("mape", mape)

			# Save the trained model
			timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
			mlflow.sklearn.log_model(self.model, f"model_{self.model_type}_{timestamp}")
			mlflow.set_tag("timestamp", timestamp)

	def predict(self, text, genre_ids):
		# Preprocess the input data (generate BERT embeddings)
		text_embedding = self.preprocess_text(text)
		genre_ids_encoded = self.mlb.transform([genre_ids])
		X = np.hstack((text_embedding.reshape(1, -1), genre_ids_encoded))

		# Normalize the features before prediction
		scaler = StandardScaler()
		X_scaled = scaler.transform(X)

		# Make prediction
		return self.model.predict(X_scaled)[0]

	def save_to_mlflow(self, model_name="movie_rating_predictor"):
		# Save MultiLabelBinarizer to a file (artifacts)
		mlb_path = "mlb.pkl"
		with open(mlb_path, 'wb') as f:
			pickle.dump(self.mlb, f)

		# Save the trained model (e.g., sklearn model)
		model_path = "model.pkl"
		with open(model_path, 'wb') as f:
			pickle.dump(self.model, f)

		artifacts = {
			'mlb': mlb_path,   # Path to the MultiLabelBinarizer
			'model': model_path,  # Path to the trained model
		}

		# Define a custom PythonModel class to properly load the artifacts
		class MovieRatingPyFuncModel(mlflow.pyfunc.PythonModel):
			def load_context(self, context):
				# Load MultiLabelBinarizer
				with open(context.artifacts['mlb'], 'rb') as f:
					self.mlb = pickle.load(f)

				# Load the trained model (pickle)
				with open(context.artifacts['model'], 'rb') as f:
					self.model = pickle.load(f)

			def predict(self, context, model_input):
				text, genre_ids = model_input
				# Preprocess the input (generate BERT embeddings)
				text_embedding = self.preprocess_text(text)
				genre_ids_encoded = self.mlb.transform([genre_ids])
				X = np.hstack((text_embedding.reshape(1, -1), genre_ids_encoded))

				# Predict using the loaded model
				return self.model.predict(X)[0]

		# Log the model to MLflow
		mlflow.pyfunc.log_model(
			artifact_path="model",
			python_model=MovieRatingPyFuncModel(),
			artifacts=artifacts,
			registered_model_name=model_name
		)


	def train(self):
		mlflow.end_run()
		try:
			with mlflow.start_run(run_name=f"train_{self.model_type}"):

				# Train and evaluate the main model
				self.train_and_evaluate("overview", "vote_average")

				# Save the entire predictor instance to MLflow
				self.save_to_mlflow()

		except Exception as e:
			logging.error(f"Error during training {self.model_type}: {e}")
		

	def get_data(self):
		self.bucket_name = os.getenv("S3_BUCKET")
		self.bucket_raw_movie_folder = os.getenv("S3_RAW_MOVIE_FOLDER")
		self.bucket_silver_movie_folder = os.getenv("S3_SILVER_MOVIE_FOLDER")
		self.bucket_gold_model_movie_folder = os.getenv("S3_GOLD_MODEL_MOVIE_FOLDER")
		self.bucket_endpoint= os.getenv("S3_ENDPOINT")
		self.bucket_access_key = os.getenv("S3_ACCESS_KEY")
		self.bucket_secret_key = os.getenv("S3_SECRET_KEY")
		self.bucket_use_ssl = os.getenv("S3_USE_SSL")
		
		self.conn = duckdb.connect(config = {'threads': 5})
		
		self.conn.execute(f"""
			DROP SECRET IF EXISTS s3secret;
			CREATE SECRET s3secret (
				TYPE S3,
				KEY_ID '{self.bucket_access_key}',
				SECRET '{self.bucket_secret_key}',
				REGION 'us-east-1',
				ENDPOINT '{self.bucket_endpoint}',
				URL_STYLE 'path',
				USE_SSL '{self.bucket_use_ssl}'
			);  
		""")		

		self.conn.execute(f"""
					CREATE TABLE movies AS (
					SELECT 
						id
						, genre_ids
						, title
						, vote_average
						, overview
					FROM read_parquet('{self.bucket_gold_model_movie_folder}', hive_partitioning = true)
					);				 
					""")

		#df = conn.execute(f"SELECT genre_ids, title, vote_average FROM read_json('{output_movies_folder}/*.json')").fetchdf()

		self.input_df = self.conn.execute(f"""
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

		logging.info(f"number of lines: {self.input_df.count()}")

		return self.input_df


	def run_all(self):
		self.get_data()
		self.train()
		

MovieRatingPredictor(model_type_s="mlp").run_all()