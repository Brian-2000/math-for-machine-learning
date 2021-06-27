import statistics as s
from sklearn import datasets
import pandas as pd
import seaborn as sms
import matplotlib.pyplot as plt

my_data = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 100, 200, 300, 2, 3, 2, 2, 2, 2, 2, 2, 2]

print('The mean, Median and Mode are: ')
print(s.mean(my_data))
print(s.median(my_data))
print(s.mode(my_data))

print('The variance and standard Deviation are: ')
print(s.pvariance(my_data))
print(s.stdev(my_data))

'''
iris = datasets.load_iris()
data = pd.DataFrame(iris['data'], columns=['petal length', 'petal width',
                                            'sepal length', 'Sepal Width'])
data['species'] = iris['target']
data[''species] = data['species'].apply(lambda x: iris['target_names'][x])
print(data.describe())
'''