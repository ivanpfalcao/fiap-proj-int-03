import requests
import os 
import json
import time

number_of_pages = 200
tmdb_token = os.getenv("TMDB_TOKEN")
output_movies_folder = os.getenv("OUTPUT_MOVIES_FOLDER")

output_file_path = os.path.join(output_movies_folder,'movies_list_2.json')
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {tmdb_token}"
}

movies_list = []

with open(f'{output_file_path}', 'w') as output_file:
    for page_id in range(1, number_of_pages):
        url = f"https://api.themoviedb.org/3/movie/top_rated?language=en-US&page={page_id}"

        response = requests.get(url, headers=headers)

        movies_list = movies_list + json.loads(response.text)['results']

        for movie in movies_list:
            if not movie['adult']:
                output_file.write(json.dumps(movie) + '\n')

        time.sleep(0.5)


    



