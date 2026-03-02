import datetime as dt

class my_logger():
    def __init__(self, file_name):
        self.file_name = file_name

    def log(self, message):
        f" {dt.datetime.now()} --- " += message + "\n"
        
        with open(self.file_name, 'a') as f:
            f.write(message)
