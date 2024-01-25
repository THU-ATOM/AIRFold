from pathlib import Path


def get_module_path(module):
    return Path(module.__file__).resolve()