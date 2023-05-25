import os
for filename in os.listdir("result"):
    print(filename)
    with open("result_deepfool/{}".format(filename), "r") as f :
        lines = f.readlines()
        i = 0
        tem = []
        for line in lines:
            if i % 5 != 0:
                index = line.split(":")[1].replace("\n", "")
                index = float(index)
                tem.append(index)
            else:
                if tem != []:
                    print(tem[0], tem[1], tem[2], tem[3])
                tem = []    
            i += 1
        if tem != []:
            print(tem[0], tem[1], tem[2], tem[3])
            
