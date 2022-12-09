import sys
import os
sys.path.append('..')

from settings import PROJECT_ROOT, DATA_DIR

def check_make_data_dir(root, data_dir):
	if os.path.isdir(os.path.join(root, data_dir)) is False:
		os.makedirs(os.path.join(root, data_dir))
	# else:
		# print('Data directory exists')
	pass
	
if __name__ == '__main__':
	check_make_data_dir(PROJECT_ROOT, DATA_DIR)