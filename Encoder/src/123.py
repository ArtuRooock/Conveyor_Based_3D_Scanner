s=0
A=[2, 4, 6, 1, 7, 2, 3, 6, 7, 2]
for i in range(0,9):
    if A[i]<A[i+1]:
        A[i+1] -= A[i]
    else:
        A[i] -= A[i+1]
    s+=A[i]
print(s)