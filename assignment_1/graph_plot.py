import numpy as np
import matplotlib.pyplot as plt

peak_gflops = 200.0 
bandwidth_gbs = 30.0  

measurements_with_ai = {
    
    "dp1_1e6": {"performance": 1314964516.853 / 1e9, "ai": 1314964516.853 / (5.260e9), "desc": "Basic C loop", "bandwidth": 5.260},
    
    "dp1_3e8": {"performance": 1291153997.743 / 1e9, "ai": 1291153997.743 / (5.165e9), "desc": "Basic C loop", "bandwidth": 5.165},
    
    
    "dp2_1e6": {"performance": 4217381202.803 / 1e9, "ai": 4217381202.803 / (16.870e9), "desc": "Unrolled C loop", "bandwidth": 16.870},
    
    "dp2_3e8": {"performance": 2571879969.798 / 1e9, "ai": 2571879969.798 / (10.288e9), "desc": "Unrolled C loop", "bandwidth": 10.288},
    
    
    "dp3_1e6": {"performance": 23125787331.883 / 1e9, "ai": 23125787331.883 / (92.503e9), "desc": "Intel MKL BLAS", "bandwidth": 92.503},
    
    "dp3_3e8": {"performance": 11464605272.551 / 1e9, "ai": 11464605272.551 / (45.858e9), "desc": "Intel MKL BLAS", "bandwidth": 45.858},
    
    
    "dp4_1e3": {"performance": 6818237.914 / 1e9, "ai": 6818237.914 / (0.025e9), "desc": "Python naive", "bandwidth": 0.025},
    
    "dp4_1e4": {"performance": 6907267.802 / 1e9, "ai": 6907267.802 / (0.026e9), "desc": "Python naive", "bandwidth": 0.026},
    
    
    "dp5_1e6": {"performance": 6105010912.904 / 1e9, "ai": 6105010912.904 / (22.743e9), "desc": "NumPy dot", "bandwidth": 22.743},
    
    "dp5_3e8": {"performance": 3149433680.489 / 1e9, "ai": 3149433680.489 / (11.733e9), "desc": "NumPy dot", "bandwidth": 11.733},
}

labels = list(measurements_with_ai.keys())
y_vals = np.array([measurements_with_ai[k]["performance"] for k in labels])
x_vals = np.array([measurements_with_ai[k]["ai"] for k in labels])

ai = np.logspace(-3, 2, 400) 

memory_bound = bandwidth_gbs * ai

compute_bound = np.full_like(ai, peak_gflops)

roofline = np.minimum(memory_bound, compute_bound)

plt.figure(figsize=(12, 8))

plt.loglog(ai, roofline, 'b-', linewidth=3, label='Roofline Model')

plt.loglog(ai, memory_bound, 'r--', linewidth=2, alpha=0.7, label=f'Memory Bound ({bandwidth_gbs} GB/s)')
plt.loglog(ai, compute_bound, 'g--', linewidth=2, alpha=0.7, label=f'Compute Bound ({peak_gflops} GFLOP/s)')

plt.axvline(0.25, color='orange', linestyle='-.', linewidth=1, alpha=0.7, 
            label='Dot Product AI = 0.25 FLOP/byte')

plt.scatter(x_vals, y_vals, color='red', marker='o', s=100, zorder=5, 
            edgecolors='black', linewidth=1, label='Measurements')

# Add labels for each measurement point with better spacing
for i, txt in enumerate(labels):
    desc = measurements_with_ai[txt]["desc"]
    # Alternate label positions to avoid overlap
    offset_y = 1.1 if i % 2 == 0 else 0.9
    plt.annotate(f'{txt}\n({desc})', (x_vals[i], y_vals[i]), 
                xytext=(10, 10 if i % 2 == 0 else -25), textcoords='offset points',
                fontsize=8, ha='left' if i % 2 == 0 else 'right', 
                va='bottom' if i % 2 == 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
plt.ylabel('Performance (GFLOP/s)', fontsize=12)
plt.title('Roofline Model: Peak Performance = 200 GFLOP/s, Memory Bandwidth = 30 GB/s', 
          fontsize=14, fontweight='bold')

plt.xlim(1e-3, 1e2)
plt.ylim(1e-3, peak_gflops * 2)

plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

ridge_point = peak_gflops / bandwidth_gbs
plt.axvline(ridge_point, color='purple', linestyle=':', alpha=0.8)
plt.text(ridge_point * 1.1, peak_gflops * 0.8, 
         f'Ridge Point\nAI = {ridge_point:.2f}', 
         fontsize=10, ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

print("Analysis:")
print(f"Ridge point (transition from memory-bound to compute-bound): AI = {ridge_point:.2f} FLOP/byte")
print("\nCalculated AI from your benchmark data:")
for label in labels:
    ai_val = measurements_with_ai[label]["ai"]
    perf_val = measurements_with_ai[label]["performance"]
    bw_val = measurements_with_ai[label]["bandwidth"]
    desc = measurements_with_ai[label]["desc"]
    if ai_val < ridge_point:
        region = "memory-bound"
        expected_perf = bandwidth_gbs * ai_val
    else:
        region = "compute-bound"  
        expected_perf = peak_gflops
    efficiency = perf_val/expected_perf*100 if expected_perf > 0 else 0
    print(f"{label} ({desc}):")
    print(f"  AI = {perf_val*1e9:.0f} FLOP/s ÷ {bw_val:.1f}e9 bytes/s = {ai_val:.3f} FLOP/byte")
    print(f"  Performance: {perf_val:.3f} GFLOP/s, Bandwidth: {bw_val:.1f} GB/s")

print(f"\nKey Insights:")
print(f"- Ridge point: {ridge_point:.2f} FLOP/byte")
print(f"- ALL measured AIs ≈ 0.25-0.27 FLOP/byte - they're all essentially the same!")
print(f"- This confirms all implementations are doing the same fundamental operation")
print(f"- Performance differences come from HOW EFFICIENTLY they execute, not different AI")
print(f"- dp3 (MKL) achieves {measurements_with_ai['dp3_1e6']['bandwidth']:.1f} GB/s bandwidth vs theoretical {bandwidth_gbs} GB/s")
print(f"- This suggests MKL uses advanced techniques (vectorization, prefetching, cache optimization)")
print(f"- The roofline model parameters may need adjustment for this specific system")

print(f"\nPerformance range: {min(y_vals):.3f} - {max(y_vals):.3f} GFLOP/s")
print(f"Best performance: {max(y_vals):.3f} GFLOP/s ({labels[np.argmax(y_vals)]})")