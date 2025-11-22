FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt_tab

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('unitary/unbiased-toxic-roberta'); \
AutoModelForSequenceClassification.from_pretrained('unitary/unbiased-toxic-roberta')"

COPY . .

EXPOSE 8000

CMD [ "python", "main.py" ]
