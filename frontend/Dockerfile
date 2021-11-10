FROM python:3.8
COPY ./* ./app/
WORKDIR /app/
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install libsm6 libxrender1 libfontconfig1 libice6 libgl1 -y
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]

CMD ["frontend_app.py"]