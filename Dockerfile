# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (so Docker can cache dependencies)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy application files
COPY app.py .
COPY rf_classifier.pkl .
COPY preprocessor.pkl .
COPY Churn_Prediction_Final.csv .
COPY templates/ ./templates/

# Expose Flask's port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
