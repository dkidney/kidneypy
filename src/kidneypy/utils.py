import math

def format_bytes(x):
    c = 1024
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    for i, unit in enumerate(units):
        if x < pow(c, i + 1):
            return f"{math.floor(x / math.pow(c, i))} {unit}"
