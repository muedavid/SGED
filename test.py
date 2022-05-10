import tensorflow as tf
import Nets.backbones as backbones



a = (1,2,3)

def a(x):
    def b(t,p):
        return x*(t-p)
    return b

l = a(5)

print(l(3,1))


def a(t,p,x):
    return x * (t - p)

l = lambda t,p: a(t,p,5)
print(l(3,1))