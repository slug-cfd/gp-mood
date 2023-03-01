from sympy import *

x, y, d, l = symbols('x y d l')

numP  = (y - x)/d + 1/2
numM  = (y - x)/d - 1/2
denom = sqrt(2)*l/d

fP = erf(numP/denom)
fM = erf(numM/denom)

f = l/d*sqrt(pi/2)*(fP - fM)

simplify(diff(fP,x,x))


#((-0.5*d - x + y)*exp((0.5*d - x + y)**2/(2*l**2)) + (-0.5*d + x - y)*exp((0.5*d + x - y)**2/(2*l**2)))*exp(-((0.5*d - x + y)**2 + (0.5*d + x - y)**2)/(2*l**2))/(d*l**2)
#sqrt(2)*(-0.5*d + x - y)*exp(-(0.5*d - x + y)**2/(2*l**2))/(sqrt(pi)*l**3)
