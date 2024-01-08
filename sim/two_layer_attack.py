import numpy as np
import random
import matplotlib.pyplot as plt

normalize_flag = 1#0-basic gradient, 1-l2 normalization 2-l_infty normalization 

seed = 0
np.random.seed(seed) 
random.seed(seed) 
#two layer relu attack
repeat_time = 10
iter_step = 100
d1 = 100
d2 = 20
d3 = 10

sigma1 = 2
mu1 = 1
sigma2 = 1
mu2 = 0
eps = 0.25
if(normalize_flag == 1):
    eta = 0.05
elif(normalize_flag == 2):
    eta = 0.005
else:
    eta = 1e-3

#1-BP 2-LinBP


def forward(v,w,x):
    y = np.maximum(np.matmul(w,x),0)
    y = np.matmul(v,y)
    return y

def backward(v,w,x,x0,mod):
    y = forward(v,w,x)
    y0 = forward(v,w,x0)
    if(mod == 0):
        gra = np.matmul(np.transpose(w),np.diag(np.matmul(w,x) > 0))
        gra = np.matmul(gra, np.transpose(v))
        gra = np.matmul(gra, (y-y0))/d1
    else:
        gra = np.transpose(w)
        gra = np.matmul(gra, np.transpose(v))
        gra = np.matmul(gra,y-y0)/d1
    return gra

ttl_err1 = np.zeros([repeat_time,iter_step+1])
ttl_err2 = np.zeros([repeat_time,iter_step+1])
for times in range(repeat_time):
    W = np.random.randn(d2,d1)
    V = np.random.randn(d3,d2)
    x_star = sigma1 * np.random.randn(d1) + mu1
    x = sigma2 * np.random.randn(d1) + mu2
    x_iter1 = np.zeros([d1,iter_step+1])
    x_iter2 = np.zeros([d1,iter_step+1])

    x_iter1[:,0] = x
    x_iter2[:,0] = x
    err1 = np.zeros(iter_step+1)
    err2 = np.zeros(iter_step+1)
    x1 = x
    x2 = x
    x0 = x
    err1[0] = np.linalg.norm(x1-x_star, ord = 1)
    err2[0] = np.linalg.norm(x2-x_star, ord = 1)
    out1 = forward(V,W,x1)
    out2 = forward(V,W,x2)

    for i in range(iter_step):

        x1 = x_iter1[:,i]
        x2 = x_iter2[:,i]
        g1 = backward(V,W,x1,x_star,0)
        g2 = backward(V,W,x2,x_star,1)

        if(normalize_flag == 1):
            x1 = x1 - eta * g1 /  np.linalg.norm(g1)
        elif(normalize_flag == 2):
            x1 = x1 - eta * ((g1 > 0).astype(int) - (g1 < 0).astype(int))
        else:
            x1 = x1 - eta * g1
        temp = x1 - x0
        temp = np.clip(temp, -1 * eps, eps)
        x1 = x0 + temp
        if(normalize_flag == 1):
            x2 = x2 - eta * g2 /  np.linalg.norm(g2)
        elif(normalize_flag == 2):
            x2 = x2 - eta * ((g2 > 0).astype(int) - (g2 < 0).astype(int))
        else:
            x2 = x2 - eta * g2
        temp = x2 - x0
        temp = np.clip(temp, -1 * eps, eps)
        x2 = x0 + temp

        x_iter1[:,i+1] = x1
        x_iter2[:,i+1] = x2



        err1[i+1] = np.linalg.norm(x1-x_star, ord = 1)
        err2[i+1] = np.linalg.norm(x2-x_star, ord = 1)
        out1 = forward(V,W,x1)
        out2 = forward(V,W,x2)


    ttl_err1[times,:] = err1
    ttl_err2[times,:] = err2
    print('Number %d is finished!'%times)



x = np.arange(0,iter_step+1)
plt.plot(x,np.mean(ttl_err1,axis = 0),x,np.mean(ttl_err2,axis = 0))
plt.legend(['BP','LinBP'])
plt.show()



#np.save('err_1_l2.npy',np.mean(ttl_err1,axis = 0))
#np.save('err_2_l2.npy',np.mean(ttl_err2,axis = 0))



