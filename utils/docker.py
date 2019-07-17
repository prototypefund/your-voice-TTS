import random
import string


def generate_random_string(length=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choices(letters, k=length))


