import os


def get_docker_image_id():
    return os.environ["DOCKER_IMAGE_ID"]
