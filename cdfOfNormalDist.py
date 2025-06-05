"""
Write a function to get CDF of normal distribution
"""

"""
CDF of normal distribution =  0.5*(1 + erf(z))
 
where z = x - mean / std*(2*(0.5))

CDF = probability that a random variable X will have value <=x , if the distribution follows normal distribution
"""


import math
from scipy.special import erf


def CDF(x,mean,std):

    z = (x - mean) / (std * math.sqrt(2))

    res = 0.5*(1 + erf(z))

    return res


# Example usage
mean = 0  # Mean of the normal distribution
std_dev = 1  # Standard deviation of the normal distribution
x = 1  # Value to calculate the CDF for

result = CDF(x, mean, std_dev)
print(f"The CDF of N({mean}, {std_dev**2}) at x = {x} is: {result}")

