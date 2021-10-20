import sympy as sp

t = sp.symbols('t')
x = sp.Function('x')

diffeq = sp.Eq(x(t).diff(t),  - 0.2*x(t) - 2*(x(t)**2)*sp.cos(2*t))         # sp.diff(x(t), t)  + 0.2*x + 2*cos(2*t)*x*x
#res = sp.dsolve(diffeq, ics={x(0):1})
res = sp.dsolve(diffeq)
print(res)
#   1 - 2.2*t + 4.62*t**2 - 8.36133*t**3 + 14.5434*t**4 - 24.47283*t**5
#   1 - 2.2*t + 4.62*t**2 - 8.36133*t**3 + 14.5434*t**4 - 24.47283*t**5
#   O(t**6)