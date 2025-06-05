"""
Create a function rand10() that generates a uniform random integer between 1 and 10 (inclusive).
You can only use a given function rand7() which generates a uniform random integer between 1 and 7 (inclusive).
Your solution should be as efficient as possible.
"""
import random

def rand1_7():

    return random.randint(1,7)


def rand1_10():
    while True:
        num = (rand1_7()-1) * 7 + rand1_7()

        if num <= 40:
            return (num-1)%10 + 1


print(rand1_10())