# 과제3 loss, r2출력

from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

# B = 흑인의 비율
# input 13, output 1(506)
 
 # 완료하시오.