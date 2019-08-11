import os
import datetime
import time


# Create a directories in the path if they dont exist
def create_directory_if_not_exist(path):

    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():

    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
