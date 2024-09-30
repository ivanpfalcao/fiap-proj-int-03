import streamlit as st
import mlflow
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# List of 16 genres that were likely used during training (from your full genre list)
GENRES = {
	"Action": 28,
	"Adventure": 12,
	"Animation": 16,
	"Comedy": 35,
	"Crime": 80,
	"Drama": 18,
	"Family": 10751,
	"Fantasy": 14,
	"History": 36,
	"Horror": 27,
	"Mystery": 9648,
	"Romance": 10749,
	"Science Fiction": 878,
	"Thriller": 53,
	"War": 10752,
	"Western": 37
}

# Load the model artifact manually after getting the MLflow model URI
@st.cache_resource
def load_model_once():
	try:
		# Set the tracking URI if necessary
		mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
		
		# Specify the model URI directly from the MLflow run
		logged_model = 'runs:/bb8fe78c579e4390948a5f490703ac29/model_mlp_2024-09-30_02-03-32'

		# Load the MLflow model
		model = mlflow.pyfunc.load_model(logged_model)
		
		st.success("Model loaded successfully from MLflow.")
		return model
	except Exception as e:
		st.error(f"Error loading model from MLflow: {e}")
		return None

# Load BERT model for generating text embeddings
@st.cache_resource
def load_bert_model():
	return SentenceTransformer('all-MPNet-base-v2')  # Using MPNet model

# Predict function
def predict(model, description, genre_ids):
	try:
		# Load the BERT model to generate embeddings for the description
		bert_model = load_bert_model()

		# Generate embeddings for the description (text)
		description_embedding = bert_model.encode(description, convert_to_tensor=False)

		# Number of possible genres (based on the provided genre list)
		possible_genres = len(GENRES)

		# Convert genre_ids to a multi-hot encoding
		genre_encoding = np.zeros(possible_genres)
		for genre_id in genre_ids:
			# Find the index of the genre_id and mark it in the multi-hot encoding
			genre_index = list(GENRES.values()).index(genre_id)
			genre_encoding[genre_index] = 1

		# Combine description_embedding (768 features) and genre_encoding (16 features)
		combined_input = np.hstack((description_embedding, genre_encoding))

		# Reshape to match the input shape expected by the model
		combined_input = combined_input.reshape(1, -1)  # Ensure it's a 2D array with 1 row

		# Make prediction using the model
		prediction = model.predict(combined_input)
		return prediction
	except Exception as e:
		st.error(f"Error during prediction: {e}")
		return None

# Streamlit UI
def main():
	st.title("Movie Rating Predictor")
	st.write("This app predicts movie ratings based on the description using a model loaded from MLflow.")

	# Load the model when the app starts (it will stay cached in memory)
	model = load_model_once()

	# Input fields for the movie description
	description = st.text_area("Enter the movie description:")

	# Multi-select box for genres (restricting to the 16 genres used during training)
	selected_genres = st.multiselect(
		"Select the genres for the movie:",
		options=list(GENRES.keys())  # Show genre names as options
	)

	# Map the selected genre names to their corresponding IDs
	genre_ids = [GENRES[genre] for genre in selected_genres]

	# Predict button
	if st.button("Predict"):
		if description and genre_ids:
			try:
				if model:
					# Make prediction
					rating = predict(model, description, genre_ids)

					if rating is not None:
						# Show the predicted rating
						st.success(f"The predicted rating for the movie is: {rating[0]:.2f}")
			except ValueError:
				st.error("There was an error during prediction.")
		else:
			st.warning("Please enter a movie description and select at least one genre to predict the rating.")

if __name__ == "__main__":
	main()
