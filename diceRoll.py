"""
Dice Roll Probability: Given two dice, calculate the probability of rolling a sum greater than a given number
𝑘

"""

#k = sum; n : No of dices
def dice_sum(k,n):

    fav_outcomes = 0
    total_outcomes = 6**n

    for i in range(1,7):
        for j in range(1,7):

            curr_sum = (i+j)

            if curr_sum > k:
                fav_outcomes+=1

    return (fav_outcomes/total_outcomes)

#Note:  we can make this more efficient by  using sliding window = 2, and sum = 6 and treating the dice values as 2 arrays.
if __name__ == "__main__":

    print(f" prob of sum >  10 : {dice_sum(10,2)}")