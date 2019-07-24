from D4M.assoc import *

# Instantiate test data

rows = list()
cols = list()
vals = list()

with open("benchmarking_rows.txt") as out_file:
    for line in out_file:
        line = line.rstrip('\n')
        rows.append(line)

with open("benchmarking_cols.txt",'r') as out_file:
    for line in out_file:
        line = line.rstrip('\n')
        cols.append(line)
        
with open("benchmarking_vals.txt",'r') as out_file:
    for line in out_file:
        line = line.rstrip('\n')
        vals.append(line)
        
# Run Benchmark
        
n = 2**np.arange(5,19)

K=8

MaxGB = 2
MaxGF = 4*1.6

m = K * n
assoc_gbytes = np.zeros(np.size(n))
assoc_flops = np.zeros(np.size(n))
assoc_gflops = np.zeros(np.size(n))
assoc_time = np.zeros(np.size(n))

for i in range(np.size(n)):
    for j in range(10):
        start = time.time()    
        A = Assoc(rows[i],cols[i],vals[i])
        stop = time.time()

        assoc_time[i] += stop-start
        assoc_flops[i] = len(vals[i])
        print(assoc_flops[i])
        ii,jj,vv = A.find()
        assoc_gbytes[i] += (len(ii)+len(jj)+ 8*m[i])/1e9
        assoc_gflops[i] += assoc_flops[i]/(stop-start)/1e9
    
    assoc_time[i] = assoc_time[i]/10
    assoc_gbytes[i] = assoc_gbytes[i]/10
    assoc_gflops[i] = assoc_gflops[i]/10
    
    print("Time: ", assoc_time[i], end='')
    print(", GFlops: ", assoc_gflops[i], end='')
    print(", GBytes: ", assoc_gbytes[i])
    
print("const_py_time = ",list(assoc_time))

print("const_py_gbytes = ",list(assoc_gbytes))

print("const_py_gflops = ",list(assoc_gflops))

