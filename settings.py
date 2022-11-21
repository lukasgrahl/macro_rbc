import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def check_make_data_dir(root, data_dir):
	if os.path.isdir(os.path.join(root, data_dir)) is False:
		os.makedirs(os.path.join(root, data_dir))
	pass

if __name__ == '__main__':
	check_make_data_dir(PROJECT_ROOT, DATA_DIR)