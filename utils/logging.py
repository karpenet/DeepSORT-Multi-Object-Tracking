import logging

class Logger:
    def __init__(self, log_file: str = 'log.txt', level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.log_file = log_file

        # Create a file handler for logging
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(level)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

    def log(self, message: str) -> None:
        self.logger.info(message)

    def error(self, message: str) -> None:
        self.logger.error(message)


# Example usage:
if __name__ == "__main__":
    log = Logger()

    log.info("This is an info message.")
    log.error("This is an error message.")

