import logging
import os
from datetime import datetime

# Create log directory path
log_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_path, exist_ok=True)

# Generate log file name based on current timestamp
log_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create full log file path
log_file_path = os.path.join(log_path, log_filename)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
    level=logging.INFO  # Set logging level to INFO
)

# Test logging
logging.info("Logging system configured successfully")

__all__ = ['logging']