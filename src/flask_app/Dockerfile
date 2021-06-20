FROM python:3.7.3

# Set working directory
WORKDIR /app

# Copy all files from flask-app directory to image
COPY . /app

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
