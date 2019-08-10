import numpy as np
from scipy.stats import t
from math import *


a=[-3,-1,0,1,3]
b=[5,7,9,11,13]


def t_test(x,y,var=True,tail=2):
    #insert 2 data; var and tail are default as equal and two, respectively
    nx = len(x)
    ny = len(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = (np.std(x)**2) * nx / (nx - 1)
    y_var = (np.std(y)**2) * ny / (ny - 1)
    
    #df and t-score for same variance
    if var == True :
        df = nx + ny - 2
        pool_var = ((nx - 1) * x_var + (ny - 1) * y_var) / df 
        t_score = (x_mean - y_mean) / sqrt(pool_var * (1 / nx + 1 / ny))
    #df and t-score for different variance (known as Welch's t-test)
    else:
        df = (x_var / nx + y_var / ny)**2 / ((x_var / nx)**2 / (nx-1) + (y_var / ny)**2 / (ny-1))
        t_score = (x_mean - y_mean) / sqrt(x_var / nx + y_var / ny)

    #left area and area for t distribution given t_score
    l_area = t.cdf(t_score,df)
    if(t_score > 0):
        area = 1 - l_area
    else:
        area = l_area

    #return p_value for tail=1(single) or 2(two)
    p_value = area*tail
    return t_score, p_value

print(t_test(a,b,False))
