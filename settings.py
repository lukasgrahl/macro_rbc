import os

RANDOM_SEED = 101
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJ_ROOT, "data")


def make_check_data_dir(root: str, data_dir: str):
    if os.path.isdir(os.path.join(root, data_dir)) is False:
        os.makedirs(os.path.join(root, data_dir))
    pass


if __name__ == "__main__":
    make_check_data_dir(PROJ_ROOT, DATA_DIR)
