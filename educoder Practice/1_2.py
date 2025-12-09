import numpy as np
matrix=np.ones((3,2),dtype=int) #matrix为3行2列的数组
for i in range(0,3):
    for j in range(0,2):
       matrix[i][j]=input()
       
print(matrix)
        
#任务1：输出matrix第二列的最大值
# ********** Begin *********#
max=np.max(matrix[:,1])
print(max)
# ********** End **********#

#任务2：输出matrix按行求和的结果
# ********** Begin *********#
sum=np.sum(matrix,axis=1)  
print(sum)
# ********** End **********#
