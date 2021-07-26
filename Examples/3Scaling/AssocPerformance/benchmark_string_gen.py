from D4M.assoc import string_gen
import numpy as np

n = 2 ** np.arange(5, 19)
K = 8
num = 10

with open('benchmarking_string_vals.txt', 'w') as f:
    for size_index in range(len(n)):
        for row_num in range(num):
            new_row = ''
            for row_index in range(n[size_index]):
                new_row += string_gen(K) + ','
            new_row += '\n'
            f.write(new_row)
