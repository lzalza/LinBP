import numpy as np
import random
import matplotlib.pyplot as plt

#one layer relu train GD
repeat_time = 10
iter_step = 10000
N=100
d=10

sigma1 = 3
mu1 = 1
sigma2 = 1
mu2 = 0
eta = 1e-3
#1-BP 2-LinBP


def forward(x,w):
    y = np.maximum(np.matmul(x,w),0)
    return y

def backward(x,w,w0,mod):
    y = forward(x,w)
    y0 = forward(x,w0)
    N = np.shape(x)[0]
    if(mod == 0):
        gra = np.matmul(np.transpose(x),np.diag(np.matmul(x,w) > 0))
        gra = np.matmul(gra, (y-y0))/N
    else:
        gra = np.matmul(np.transpose(x),y-y0)/N
    return gra


ttl_loss1 = np.zeros([repeat_time,iter_step+1])
ttl_loss2 = np.zeros([repeat_time,iter_step+1])
ttl_err1 = np.zeros([repeat_time,iter_step+1])
ttl_err2 = np.zeros([repeat_time,iter_step+1])
for times in range(repeat_time):
    x = np.random.randn(N,d)
    w_star = sigma1 * np.random.randn(d) + mu1
    w = sigma2 * np.random.randn(d) + mu2
    w_iter1 = np.zeros([d,iter_step+1])
    w_iter2 = np.zeros([d,iter_step+1])

    w_iter1[:,0] = w
    w_iter2[:,0] = w
    err1 = np.zeros(iter_step+1)
    err2 = np.zeros(iter_step+1)
    los1 = np.zeros(iter_step+1)
    los2 = np.zeros(iter_step+1)
    w1 = w
    w2 = w
    target = forward(x,w_star)
    err1[0] = np.linalg.norm(w1-w_star, ord = 1)
    err2[0] = np.linalg.norm(w2-w_star, ord = 1)
    out1 = forward(x,w1)
    out2 = forward(x,w2)
    los1[0] = 0.5 * np.linalg.norm(out1 - target, ord = 2)**2 / N
    los2[0] = 0.5 * np.linalg.norm(out2 - target, ord = 2)**2 / N
    for i in range(iter_step):
        x = np.random.randn(N,d)
        target = forward(x,w_star)

        w1 = w_iter1[:,i]
        w2 = w_iter2[:,i]
        g1 = backward(x,w1,w_star,0)
        g2 = backward(x,w2,w_star,1)
        w1 = w1 - eta * g1
        w2 = w2 - eta * g2

        w_iter1[:,i+1] = w1
        w_iter2[:,i+1] = w2


        err1[i+1] = np.linalg.norm(w1-w_star, ord = 1)
        err2[i+1] = np.linalg.norm(w2-w_star, ord = 1)
        out1 = forward(x,w1)
        out2 = forward(x,w2)
        los1[i+1] = 0.5 * np.linalg.norm(out1 - target, ord = 2)**2 / N
        los2[i+1] = 0.5 * np.linalg.norm(out2 - target, ord = 2)**2 / N
    ttl_loss1[times,:] = los1
    ttl_loss2[times,:] = los2
    ttl_err1[times,:] = err1
    ttl_err2[times,:] = err2
    print('Number %d is finished!'%times)


x = np.arange(0,iter_step+1)
plt.plot(x,np.mean(ttl_err1,axis = 0),x,np.mean(ttl_err2,axis = 0))
plt.show()

plt.plot(x,np.mean(ttl_loss1,axis = 0),x,np.mean(ttl_loss2,axis = 0))
plt.show()
'''
print(np.mean(ttl_err1,axis = 0))
print(np.mean(ttl_err2,axis = 0))
print(np.mean(ttl_loss1,axis = 0))
print(np.mean(ttl_loss2,axis = 0))

np.save('err_1.npy',np.mean(ttl_err1,axis = 0))
np.save('err_2.npy',np.mean(ttl_err2,axis = 0))
np.save('loss_1.npy',np.mean(ttl_loss1,axis = 0))
np.save('loss_2.npy',np.mean(ttl_loss2,axis = 0))
'''

