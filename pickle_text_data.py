import sys
from utils import TextFilePreprocessing

src_data_path = sys.argv[1]
out_data_path = sys.argv[2]

tfp = TextFilePreprocessing(filepath_in=src_data_path, filepath_out=out_data_path)
tfp.preprocess()