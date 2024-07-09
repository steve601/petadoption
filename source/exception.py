import sys
from source.logger import logging

def error_msg_details(error,error_deatail:sys):
    # returns 3 information,but we're only intrested in the last one
    _,_,exc_tb = error_deatail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    msg = str(error)
    error_msg = f'Error occured in python script name [{file_name}] line number [{line_number}] error message [{msg}]'
    return error_msg
    
class UserException(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_message = error_msg_details(error_msg,error_deatail=error_detail)
        
    def __str__(self):
        return self.error_message