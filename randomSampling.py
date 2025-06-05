"""
Implement a function that randomly samples

k items from a list of

n items without replacement.
"""

import random

def random_sample(k,arr):

    random_samples = []

    for _ in range(k):
        index = random.choice(range(len(arr)))
        random_samples.append(arr[index])
        del arr[index] #will inrease time complexity
        #we can just keep track of indices we have used and not reuse them

    return random_samples


if __name__ == "__main__":
    arr = [3,6,4,3,2,9,7,53,2,34,5,6,99,100,'diksha','dev']

    print(random_sample(5,arr))