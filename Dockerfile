FROM python:3.8
WORKDIR /app
COPY requirements.txt ./requirements.txt
#COPY USE4 .
COPY app.py .
RUN pip3 install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
