# Use the official Python 3.8 slim-buster image as the base image
FROM python:3.8-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy all files and directories from the current directory into the container's working directory
COPY . /app

# Install Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Specify the command to run when the container starts
CMD ["python3", "app.py"]

