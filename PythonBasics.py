
def pBasics(st):
    i=0
    ff = len(st)-1
    while i<ff:
        j = i+1
        f = len(st)
        while j<f:
            if st[i]==st[j]:
                return 1
            j+=1
        i+=1
    return 0
