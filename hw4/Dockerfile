FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app
RUN pip install -U pip
RUN pip install pipenv

COPY [ "model.bin", "model.bin" ]
COPY [ "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]