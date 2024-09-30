import os
import duckdb
import logging


class DataPrep():

    def __init__(self):
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

    def get_movies_raw_data(self):
        
        logging.info("Getting raw data")
        self.movies_raw_df = self.conn.execute(f"""
            CREATE TABLE raw_movies AS (
                SELECT 
                    DISTINCT
                    *
                    , substring(cast(release_date as string),1, 4) as release_year
                FROM read_json('{self.bucket_raw_movie_folder}')
            );				 
            """)
        


    def transform_movies_data(self):
        logging.info("Generating model_movies table")
        self.conn.execute(f"""
            CREATE TABLE model_movies AS (
                    SELECT
                        id
                        , genre_ids
                        , title
                        , vote_average
                        , overview
                        , release_year
                    FROM raw_movies
                    WHERE overview is not null 
                        AND trim(overview) <> ''
            );
                        
        """)

    def save_silver_movies(self):
        logging.info("Saving silver movies table")
        self.conn.execute(f"""          
            COPY raw_movies TO '{self.bucket_silver_movie_folder}' (
                FORMAT PARQUET,
                PARTITION_BY (release_year),
                OVERWRITE_OR_IGNORE true
            )
        """)

    def save_gold_model_movies(self):
        logging.info("Saving gold model_movies table")
        self.conn.execute(f"""          
            COPY model_movies TO '{self.bucket_gold_model_movie_folder}' (
                FORMAT PARQUET,
                PARTITION_BY (release_year),
                OVERWRITE_OR_IGNORE true
            )
        """)    

    def run_all(self):
        self.get_movies_raw_data()
        self.transform_movies_data()
        self.save_silver_movies()
        self.save_gold_model_movies()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DataPrep().run_all()