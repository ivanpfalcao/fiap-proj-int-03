import requests
import os 
import json
import time

number_of_pages = 100
#tmdb_token = os.getenv("TMDB_TOKEN")
#output_movies_folder = os.getenv("OUTPUT_MOVIES_FOLDER")
output_movies_folder="C:/projects/fiap-proj-int-03/app/get_movies/output_files"
tmdb_token="eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NWFlZjk4ZWVkNTM1NWRmYmUyNGVjNmZiOWU2ZGZjOCIsIm5iZiI6MTcyNjk0NTM5Mi4zMzgyOCwic3ViIjoiNjZlZjE3NzI2YzNiN2E4ZDY0OGQzYWM1Iiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.Y45bt_CPC9FnKCAPe4T2-nYnA3u48ZE6Hmm912zxzLs"
vote_greater_than=7.5
vote_lesser_than=10.0

output_file_path = os.path.join(output_movies_folder,f'movies_list_{vote_greater_than}_{vote_lesser_than}.json')
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {tmdb_token}"
}

movies_list = []

with open(f'{output_file_path}', 'w') as output_file:
    for page_id in range(1, number_of_pages):
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&sort_by=title.desc&vote_count.gte=5&vote_average.gte={vote_greater_than}&vote_average.lte={vote_lesser_than}&page={page_id}"

        response = requests.get(url, headers=headers)

        movies_list = json.loads(response.text)['results']

        for movie in movies_list:
            if not movie['adult']:
                output_file.write(json.dumps(movie) + '\n')

        time.sleep(0.5)


    



