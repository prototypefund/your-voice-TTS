import os
import random
import string


def get_docker_image_id():
    return os.environ["DOCKER_IMAGE_ID"]


def get_run_id():
    chars = string.digits + string.ascii_lowercase
    return random.choices(chars, k=7)

