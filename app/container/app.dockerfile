FROM python:3.10.15-bookworm

ENV APP_FOLDER=/app

RUN mkdir -p ${APP_FOLDER}/get_movies \
    && mkdir -p ${APP_FOLDER}/get_movies \
    && chmod 777 -R ${APP_FOLDER}

COPY ./prepare_data/requirements.txt ${APP_FOLDER}/

RUN pip install -r ${APP_FOLDER}/requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MPNet-base-v2')"


COPY ./frontend/* ${APP_FOLDER}/frontend/


COPY ./get_movies/* ${APP_FOLDER}/get_movies/
COPY ./prepare_data/* ${APP_FOLDER}/prepare_data/

CMD ["/bin/bash", "-c", "sleep infinity"]