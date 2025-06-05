"""
Write a function to simulate tossing a coin
n times and return the probability of getting heads.
"""


"""
Probabilty of heads in n tosses : No of head/ no of tosses
"""

import random

def coin_toss(n):

    if n <= 0:
        raise ValueError("Number of tosses must be a positive integer.")

    toss_outputs = ['H','T']

    heads_count = 0

    for _ in range(n):

        toss_result = random.choice(toss_outputs)

        if toss_result == 'H':
            heads_count+=1


    return (heads_count/n)



if __name__ == "__main__":

    print(f" Prob of heads if coin tossed 20 times: {coin_toss(20)}")