import csv
import logging
import sys


def get_logger(
    level: int = logging.INFO,
    propagate: bool = False,
) -> logging.Logger:
    """Get a logger with the given name with some formatting configs."""
    logger = logging.getLogger("diffusion-benchmarks")
    logger.propagate = propagate
    logger.setLevel(level)
    if not len(logger.handlers):
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class CsvHandler:
    def __init__(self, file_name, header=None):
        self.file_name = file_name
        self.header = header
        self.file = None

    def open_file(self):
        self.file = open(self.file_name, mode="a", newline="\n", encoding="utf-8")
        self.csv_writer = csv.writer(self.file)

        if self.header:
            self.csv_writer.writerow(self.header)

        print(f"File '{self.file_name}' opened successfully for writing.")

    def write_row(self, data):
        self.csv_writer.writerow(data)

    def close_file(self):
        if self.file:
            self.file.close()
            print(f"File '{self.file_name}' closed successfully.")

    def write_header(self, data):
        self.open_file()
        self.write_row(data)
        self.close_file()

    def write_results(self, result):
        self.open_file()
        self.write_row(list(result.values()))
        self.close_file()
