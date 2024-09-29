import requests
import os 
import json
import time

number_of_pages = os.getenv("TMDB_NUMBER_OF_PAGES")
tmdb_token = os.getenv("TMDB_TOKEN")
output_movies_folder = os.getenv("OUTPUT_MOVIES_FOLDER")


intervals = [0,2.5,5,7.5]


headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {tmdb_token}"
}


for interval in intervals:

    vote_greater_than=interval
    vote_lesser_than=interval + 2.5

    print(output_movies_folder)
    output_file_path = os.path.join(output_movies_folder,f'movies_list_{vote_greater_than}_{vote_lesser_than}.json')

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




    



