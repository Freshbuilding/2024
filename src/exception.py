import traceback  # To properly handle and access traceback information
import sys
import logging


def error_message_detail(error, error_detail):
    """
    Constructs an error message detailing the file, line number, and error message.
    :param error: The error that occurred.
    :param error_detail: An exception object to extract detailed information from.
    :return: A formatted error message string.
    """
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{error}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        """
        Initializes the CustomException with an error message and details.
        :param error_message: The error message.
        :param error_detail: Detailed information about the error, used to construct the detailed error message.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        """
        String representation of the exception.
        :return: The detailed error message.
        """
        return self.error_message
