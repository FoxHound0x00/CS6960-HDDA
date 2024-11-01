import math

def johnson_lindenstrauss(n, epsilon=None, delta=None, m=None):
    if epsilon is not None and delta is not None:
        m = int((2 * math.log(n / delta)) / (epsilon))
        return m
    
    elif m is not None and delta is not None:
        epsilon = (2 * math.log(n / delta)) / m
        return 1 + epsilon
    
    else:
        raise ValueError("Insufficient parameters provided. Either epsilon or m must be given.")

# For question (a): n = 1000, epsilon = 0.1, delta = 0.05
print("Target dimension m for (a):", johnson_lindenstrauss(n=1000, epsilon=0.1, delta=0.05))

# For question (b): n = 100000, epsilon = 0.1, delta = 0.05
print("Target dimension m for (b):", johnson_lindenstrauss(n=100000, epsilon=0.1, delta=0.05))

# For question (c): n = 10000, delta = 0.1, m = 1000
print("Achievable distortion for (c):", johnson_lindenstrauss(n=10000, delta=0.1, m=1000))

# For question (d): n = 10000, delta = 0.1, m = 10000
print("Achievable distortion for (d):", johnson_lindenstrauss(n=10000, delta=0.1, m=10000))
