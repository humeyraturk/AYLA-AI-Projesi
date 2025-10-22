# Python 3.10 taban imajını kullan
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Önce gereksinimleri kopyala ve kur (cache'leme için)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını kopyala
COPY . .

# FAISS indeksini ilk çalıştırmada oluştur (Bu adım önemlidir!)
# Bu, 'setup_vector_db' fonksiyonunu çalıştıracak ve indeksi oluşturacak
RUN python -c "from chatbot import setup_vector_db; setup_vector_db()"

# Flask portunu (5000) dışarıya aç
EXPOSE 5000

# Uygulamayı Gunicorn (production server) ile başlat
# Not: requirements.txt dosyana 'gunicorn' eklemeyi unutma!
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "chatbot:app"]