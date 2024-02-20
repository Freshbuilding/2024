from datetime import datetime
import os
import logging

# Create a log file name with the current timestamp
log_file_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Construct the logs path. Using 'os.path.join' ensures compatibility across operating systems
logs_path = os.path.join(os.getcwd(), "logs")

# Ensure the 'logs' directory exists
os.makedirs(logs_path, exist_ok=True)

# Corrected the LOG_FILE_PATH to point to the directory, not including the file name in the directory creation step
log_file_path = os.path.join(logs_path, log_file_name)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
