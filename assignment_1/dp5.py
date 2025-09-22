import numpy as np
import sys
import time

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} N repetitions")
    sys.exit(1)

N = int(sys.argv[1])
reps = int(sys.argv[2])

A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)


def main():
    times = []
    
    for _ in range(reps):
        t0 = time.perf_counter()
        R = np.dot(A,B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    second_half_start = reps // 2
    second_half_times = times[second_half_start:]
    time_per_run = np.mean(second_half_times)
    
    bytes_transferred = N * 8 
    bandwidth_gb_per_sec = (bytes_transferred / (1024**3)) / time_per_run
    
    # FLOPS calculation
    flops = 2 * N
    flops_per_sec = flops / time_per_run
    
    print(f"N: {N} <T>: {time_per_run:.7f} sec B: {bandwidth_gb_per_sec:.3f} GB/sec F: {flops_per_sec:.3f} FLOP/sec")
    print(f"Dot product: {R:.9f}")

if __name__ == "__main__":
    main()