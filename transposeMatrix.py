def transpose(M):

    return [list[i] for i in zip(*M)]


print(transpose([[1,2],[3,4],[5,6]]))