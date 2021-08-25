from D4M.assoc import string_gen
import numpy as np

n = 2 ** np.arange(5, 19)
K = 8
num = 10

filenames = ['benchmarking_string_rows2.txt', 'benchmarking_string_cols2.txt', 'benchmarking_string_vals2.txt']
# 'benchmarking_string_rows.txt', 'benchmarking_string_cols.txt', 'benchmarking_string_vals.txt',

for filename in filenames:
    with open(filename, 'w') as f:
        for size_index in range(len(n)):
            for row_num in range(num):
                new_row = ''
                for row_index in range(n[size_index]):
                    new_row += string_gen(K) + ','
                new_row += '\n'
                f.write(new_row)
    print('Finished writing file: ' + filename)
