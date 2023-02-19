#Метод Эйлера. Задача Коши
# x: [a,b]

import math
import numpy as np
#print(dir(math))
pi=math.pi

a=0.0
b=2.0*pi
n=101  # кол-во узлов
h=float((b-a)/(n-1))
print("h=",h)
print("h =%8.5f" %h)

def F1(x,y1,y2):
    return y2

def F2(x,y1,y2):
    return -y1

def euler2():
    for i in range(1,n-1):
        i1=i-1;
        x=xarray[i-1];
        y1array[i]=y1array[i1]+h*F1(x,y1array[i1],y2array[i1])
        y2array[i]=y2array[i1]+h*F2(x,y1array[i1],y2array[i1])
    return 0

xarray=np.arange(a,b+0.0001,h)
print(len(xarray))
print(xarray[0])

y1array=np.zeros(n)
y2array=np.zeros(n)
y1array[0]=0.0
y2array[0]=1.0

dummy=euler2()
print(y1array[0])

import matplotlib.pyplot as plt

step=2
x = xarray[0:n:step]
y1=y1array[0:n:step]
nn=len(y1)
print("len(y1)=",nn)

sline1=""
str="="
for j in np.arange(0,50):
    sline1=sline1+str
print(sline1)
str0="    x"
str1="      y1(x)"
str2="     sin(x)"
str3="   y1(x)-sin(x)"
print('{0:8s}{1:14s}{2:14s}{3:14s}'.format(str0,str1,str2,str3))
print(sline1)
for i in np.arange(0,nn):
    xi=x[i]
    y1i=y1[i]
    ares=np.sin(xi)
    print('{0:8.5f}{1:14.5e}{2:14.5e}{3:14.5e}'.format(xi,y1i,ares,y1i-ares))

xx  = np.linspace(a,b,201)
yy1 = np.sin(xx)

#create plot
#plt.figure(1, figsize = (12,8) )
#plt.plot(xx,yy1,color="blue",linestyle="solid",linewidth=5,label='analytic')
plt.plot(xx,yy1,'b-', linewidth=5,label='analytic')
plt.plot(x, y1, 'ro',label="numeric")
plt.xlabel('X')
plt.ylabel('Y1(X)')
plt.legend(loc='upper right')
plt.axhline(color='gray',zorder=-1)
plt.axvline(color='gray',zorder=-1)
plt.grid(True)

plt.gcf().set_size_inches(10, 6)
# save plot to file
plt.savefig('y1.pdf',dpi=300)
plt.savefig('y1.eps',dpi=300)

plt.show()