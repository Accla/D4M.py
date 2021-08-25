from D4M.assoc import *
import D4M.util as util
import time
import matplotlib.pyplot as plt

num = 10  # Number of lines of equal length
comparison = True  # Whether two functions are being compared against each other
numerical = True  # Whether data is numerical or not (determines if conversion from strings to numbers occurs)

# Instantiate test data

rows, cols, vals = list(), list(), list()
rows2, cols2, vals2 = list(), list(), list()

if numerical:
    row_filename = "benchmarking_num_rows.txt"
    col_filename = "benchmarking_num_cols.txt"
    val_filename = "benchmarking_num_vals.txt"
    row2_filename = "benchmarking_num_rows2.txt"
    col2_filename = "benchmarking_num_cols2.txt"
    val2_filename = "benchmarking_num_vals2.txt"
else:
    row_filename = "benchmarking_string_rows.txt"
    col_filename = "benchmarking_string_cols.txt"
    val_filename = "benchmarking_string_vals.txt"
    row2_filename = "benchmarking_string_rows2.txt"
    col2_filename = "benchmarking_string_cols2.txt"
    val2_filename = "benchmarking_string_vals2.txt"

filename_dict = {row_filename: rows, col_filename: cols, val_filename: vals,
                 row2_filename: rows2, col2_filename: cols2, val2_filename: vals2}

for filename in filename_dict.keys():
    with open(filename, 'r') as out_file:
        for line in out_file:
            line = line.rstrip('\n')
            filename_dict[filename].append(line)

# Run Benchmark

n = 2 ** np.arange(5, 19)
K = 8
MaxGB = 2
MaxGF = 4 * 1.6

m = K * n
assoc_gbytes = np.zeros(np.size(n))
assoc_gbytes_alt = np.zeros(np.size(n))
assoc_flops = np.zeros(np.size(n))
assoc_flops_alt = np.zeros(np.size(n))
assoc_gflops = np.zeros(np.size(n))
assoc_gflops_alt = np.zeros(np.size(n))
assoc_time = np.zeros(np.size(n))
assoc_time_alt = np.zeros(np.size(n))

for i in range(np.size(n)):
    for j in range(num):
        A = Assoc(rows[num * i + j], cols[num * i + j], vals[num * i + j], convert_val=numerical)
        B = Assoc(rows2[num * i + j], cols2[num * i + j], vals2[num * i + j], convert_val=numerical)
        start = time.time()
        # Insert binary operation
        C = Assoc([], [], [])  # Filler
        stop = time.time()

        assoc_time[i] += stop - start
        assoc_flops[i] = len(vals[num * i + j])
        ii, jj, vv = A.find()
        assoc_gbytes[i] += (len(ii) + len(jj) + len(vv) + 8 * m[i]) / 1e9
        assoc_gflops[i] += assoc_flops[i] / (stop - start) / 1e9

        if comparison:
            A_alt = Assoc(rows[num * i + j], cols[num * i + j], vals[num * i + j], convert_val=numerical)
            B_alt = Assoc(rows2[num * i + j], cols2[num * i + j], vals2[num * i + j], convert_val=numerical)
            start_alt = time.time()
            # Insert alternative binary operation to test against
            C_alt = Assoc([], [], [])  # Filler
            stop_alt = time.time()

            if not assoc_equal(C, C_alt):
                C.printfull()
                C_alt.printfull()
                assoc_equal(C, C_alt, return_info=True)
                raise AssertionError

            assoc_time_alt[i] += stop_alt - start_alt
            assoc_flops_alt[i] = len(vals[num * i + j])
            ii_alt, jj_alt, vv_alt = A_alt.find()
            assoc_gbytes_alt[i] += (len(ii_alt) + len(jj_alt) + len(vv_alt) + 8 * m[i]) / 1e9
            assoc_gflops_alt[i] += assoc_flops_alt[i] / (stop_alt - start_alt) / 1e9

    assoc_time[i] = assoc_time[i] / num
    assoc_gbytes[i] = assoc_gbytes[i] / num
    assoc_gflops[i] = assoc_gflops[i] / num

    print("Time: " + str(assoc_time[i]) + ", GFlops: " + str(assoc_gflops[i]) + ", GBytes: " + str(assoc_gbytes[i]))

    if comparison:
        assoc_time_alt[i] = assoc_time_alt[i] / num
        assoc_gbytes_alt[i] = assoc_gbytes_alt[i] / num
        assoc_gflops_alt[i] = assoc_gflops_alt[i] / num

        print("(Alt) Time: " + str(assoc_time_alt[i]) + ", GFlops: " + str(assoc_gflops_alt[i])
              + ", GBytes: " + str(assoc_gbytes_alt[i]))

plt.subplot(131)
plt.plot(n, assoc_time, 'g-.')
plt.xlabel('Number of entries')
plt.ylabel('Runtime (s)')
plt.subplot(132)
plt.plot(n, assoc_gflops, 'g-.')
plt.xlabel('Number of entries')
plt.ylabel('gigaflops')
plt.subplot(133)
plt.plot(n, assoc_gbytes, 'g-.')
plt.xlabel('Number of entries')
plt.ylabel('gigabytes')

if comparison:
    plt.subplot(131)
    plt.plot(n, assoc_time_alt, 'b-.')
    plt.xlabel('Number of entries')
    plt.ylabel('Runtime (s)')
    plt.subplot(132)
    plt.plot(n, assoc_gflops_alt, 'b-.')
    plt.xlabel('Number of entries')
    plt.ylabel('gigaflops')
    plt.subplot(133)
    plt.plot(n, assoc_gbytes_alt, 'b-.')
    plt.xlabel('Number of entries')
    plt.ylabel('gigabytes')

plt.show()

print("const_py_time = ", list(assoc_time))
print("const_py_gbytes = ", list(assoc_gbytes))
print("const_py_gflops = ", list(assoc_gflops))

if comparison:
    print("(alt) const_py_time = ", list(assoc_time_alt))
    print("(alt) const_py_gbytes = ", list(assoc_gbytes_alt))
    print("(alt) const_py_gflops = ", list(assoc_gflops_alt))

assoc_gbytes = np.zeros(np.size(n))
assoc_gbytes_alt = np.zeros(np.size(n))
assoc_flops = np.zeros(np.size(n))
assoc_flops_alt = np.zeros(np.size(n))
assoc_gflops = np.zeros(np.size(n))
assoc_gflops_alt = np.zeros(np.size(n))
assoc_time = np.zeros(np.size(n))
assoc_time_alt = np.zeros(np.size(n))
