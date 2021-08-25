from D4M.assoc import num_string_gen
import numpy as np

n = 2 ** np.arange(5, 19)
upper_bound = 100
num = 10

filenames = ['benchmarking_num_rows.txt', 'benchmarking_num_cols.txt', 'benchmarking_num_vals.txt',
             'benchmarking_num_rows2.txt', 'benchmarking_num_cols2.txt', 'benchmarking_num_vals2.txt']

for filename in filenames:
    with open(filename, 'w') as f:
        for size_index in range(len(n)):
            for row_num in range(num):
                new_row = num_string_gen(n[size_index], upper_bound) + '\n'
                f.write(new_row)
    print('Finished writing file: ' + filename)
