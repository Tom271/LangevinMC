import numpy as np

# Increasing diagonal Gaussian specific
p = 2 # dimension
m = 1
M = 1

def theorem1(N, h):
    ''' Given number of steps N and step size h, returns upper bound on W2 distance '''
    global p, m, M
    starting_error = np.sqrt(p/m) # Assuming U=0 at minimum (valid for gaussian)
    if h <= 2/(m+M):
        return (1 - m * h)**N * starting_error + 1.65 * (M/m) * (h * p)**0.5
    else:
        return (M * h - 1)**N * starting_error + (1.65 * M * h) / (2 - M * h) * (h * p)**0.5


h = 0.01
for N in [100, 10**3, 10**4, 10**5]:
    print(theorem1(N, h))
