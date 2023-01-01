import sys
import os
sys.path.append('..')

<<<<<<< HEAD
from settings import PROJECT_ROOT, DATA_DIR
=======
from settings import DATA_DIR, PROJECT_ROOT
>>>>>>> dev

def check_make_data_dir(root, data_dir):
	if os.path.isdir(os.path.join(root, data_dir)) is False:
		os.makedirs(os.path.join(root, data_dir))
<<<<<<< HEAD
	# else:
		# print('Data directory exists')
	pass
	
=======
	else:
		print(f'DATA_DIR is existant under: {DATA_DIR}')
	pass

>>>>>>> dev
if __name__ == '__main__':
	check_make_data_dir(PROJECT_ROOT, DATA_DIR)