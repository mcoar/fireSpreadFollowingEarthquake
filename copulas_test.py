# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:25:17 2022

@author: Lab User
"""

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import numpy as np
from pytictoc import TicToc; t = TicToc()

# Sample 10000 uniformly distributed values
x = stats.uniform(0,1).rvs(1000)
# sns.distplot(x, kde=False, norm_hist=True)

# Convert to normal dist
norm = stats.distributions.norm()
x_trans = norm.ppf(x)
# sns.distplot(x_trans)

# Plot norm vs uniform
# h = sns.jointplot(x, x_trans, stat_func=None)
# h.set_axis_labels('original', 'transformed', fontsize=16)

# Plot beta vs. uniform
beta = stats.distributions.beta(a=10,b=3)
x_trans = beta.ppf(x)
# h = sns.jointplot(x,x_trans, stat_func=None)
# h.set_axis_labels('original','transformed',fontsize=16)

# Plot gumbel vs. uniform
gumbel = stats.distributions.gumbel_l()
x_trans = gumbel.ppf(x)
# h = sns.jointplot(x, x_trans, stat_func=None)
# h.set_axis_labels('original', 'transformed', fontsize=16);

# Convert gumbel to uniform with cdf
x_trans_trans = gumbel.cdf(x_trans)
# h = sns.jointplot(x_trans,x_trans_trans,stat_func=None)
# h.set_axis_labels('original', 'transformed', fontsize=16)

# Correlated multivariate normal
toggle_plot = False

corr12 = 0.7
corr3 = 0.5

n = 1500
corr1 = 0.7
corr2 = 0.5
meanvec = [0 for col in range(n)]
covmat = np.zeros((n,n))

t.tic()
for i in range(n):
    i_even = (i % 2) == 0
    for j in range(n):
        j_even = (j % 2) == 0
        if i == j:
            covmat[i][j] = 1
        elif j_even and i_even:
            covmat[i][j] = corr1
        else:
            covmat[i][j] = corr2
t.toc('Covariance Matrix generation time:')

            
t.tic()
mvnorm = stats.multivariate_normal(mean = meanvec,cov=covmat)
t.toc('Distribution generation time:')

t.tic()            
x = mvnorm.rvs(10000)
t.toc('Sample generation time:')

corr12,_ = stats.pearsonr(x[:,0],x[:,1])

corr23,_ = stats.pearsonr(x[:,1],x[:,2])

corr31,_ = stats.pearsonr(x[:,2],x[:,0])

if toggle_plot:
    h = sns.jointplot(x[:,0],x[:,1],kind='kde', stat_func=None)
    h.set_axis_labels('X1','X2',fontsize=16)
    
    
    h = sns.jointplot(x[:,1],x[:,2],kind='kde', stat_func=None)
    h.set_axis_labels('X2','X3',fontsize=16)
    
    
    h = sns.jointplot(x[:,2],x[:,0],kind='kde', stat_func=None)
    h.set_axis_labels('X3','X1',fontsize=16)


print('correlation: {:0.3f}, {:0.3f}, {:0.3f}'.format(corr12,corr23,corr31))

# Uniformify the marginals
t.tic()
norm = stats.norm()
x_unif = norm.cdf(x)
t.toc('Uniformify time:')

corr12,_ = stats.pearsonr(x_unif[:,0],x_unif[:,1])

corr23,_ = stats.pearsonr(x_unif[:,1],x_unif[:,2])

corr31,_ = stats.pearsonr(x_unif[:,2],x_unif[:,0])

print('uniform correlation: {:0.3f}, {:0.3f}, {:0.3f}'.format(corr12,corr23,corr31))

if toggle_plot:
    h = sns.jointplot(x_unif[:,0],x_unif[:,1], kind='hex',stat_func=None)
    h.set_axis_labels('Y1','Y2',fontsize=16)
    
    
    h = sns.jointplot(x_unif[:,1],x_unif[:,2], kind='hex',stat_func=None)
    h.set_axis_labels('Y2','Y3',fontsize=16)
    
    h = sns.jointplot(x_unif[:,2],x_unif[:,0], kind='hex',stat_func=None)
    h.set_axis_labels('Y3','Y2',fontsize=16)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x[:,0],x[:,1],x[:,2],c=x[:,2],cmap='Greens')


# Transform marginals into different distributions
# m1 = stats.gumbel_l()
# m2 = stats.beta(a=10,b=2)

# x1_trans = m1.ppf(x_unif[:,0])
# x2_trans = m2.ppf(x_unif[:,1])

# h = sns.jointplot(x1_trans,x2_trans,kind='kde',xlim=(-6,2),ylim=(.6,1.0),stat_func=None)
# h.set_axis_labels('Maximum river level','Probability of flooding', fontsize=16)

# Uncorrelated comparison
# x1 = m1.rvs(10000)
# x2 = m2.rvs(10000)

# h = sns.jointplot(x1,x2,kind='kde', xlim=(-6,2),ylim=(.6,1.0),stat_func=None)
# h.set_axis_labels('Maximum River Level','Probability of flooding', fontsize=16)


