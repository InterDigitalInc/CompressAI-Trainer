import os


def hostname() -> str:
    cmd = "hostname"
    return os.popen(cmd).read().rstrip()


def username() -> str:
    return os.environ["USER"]
