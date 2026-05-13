import sys
from src.healthcare_outcome_mlops.logging import logger


class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        self.error_message = self.error_message
        _, _, exc_tb = error_detail.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        self.file_name, self.lineno, str(self.error_message)

if __name__ == "__main__":
    try:
        logger.logging.info("Testing custom exception")
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)