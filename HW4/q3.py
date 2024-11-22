import math

def calculate_epsilon(n, beta, delta):
    """
    epsilon = sqrt((beta^2 * ln(1/delta)) / n)
    n : Number of samples.
    beta : Bound on the distance of points from the mean.
    delta : Failure probability.
    """
    epsilon = math.sqrt((beta**2 * math.log(1 / delta)) / n)
    return epsilon

# from q
n = 3000
beta = 2
delta = 0.1

# Calculation
epsilon = calculate_epsilon(n, beta, delta)
print(f"Guaranteed ε: {epsilon}")
