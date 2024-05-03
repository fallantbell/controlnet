
flag = False
for i in range(10):
    if i==5 and flag==False:
        i-=1
        flag = True
    print(i)