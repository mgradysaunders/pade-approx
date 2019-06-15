# Pade approximant coefficients from Taylor series coefficients.
def pade(M, N, coeff):
    import numpy as np
    import itertools as it
    if M <= N:
        C = np.zeros((N, N))
        for i, j in it.product(range(N), range(N)):
            k = M + i - j
            if k >= 0:
                C[i, j] = coeff(k)

        c = np.zeros(N)
        for k in range(N):
            c[k] = -coeff(M + k + 1)

        b = np.linalg.solve(C, c)
        b = np.array([1, *b.tolist()])
        a = np.zeros(M + 1)
        for m in range(M + 1):
            for k in range(m + 1):
                a[m] += b[m - k] * coeff(k)

        a = [*a.tolist()]
        b = [*b.tolist()]
        return a, b
    else:
        C = np.zeros((M + 1, M + 1))
        for i, j in it.product(range(M + 1), range(M + 1)):
            if j < M - N + 1:
                if i == j:
                    C[i, j] = -1
            else:
                C[i, j] = coeff(M + i - j)

        c = np.zeros(M + 1)
        for k in range(M + 1):
            c[k] = -coeff(N + k)

        w = np.linalg.solve(C, c)
        a = np.zeros(M + 1)
        b = np.zeros(N + 1)
        b[0] = 1
        for k in range(M + 1):
            if k < M - N + 1:
                a[k + N] = w[k]
            else:
                b[k - M + N] = w[k]

        for m in range(N):
            for k in range(min(m, N) + 1):
                a[m] += b[m - k] * coeff(k)

        a = a.tolist()
        b = b.tolist()
        return a, b

# Example.
import numpy as np
from scipy.special import sici
from matplotlib import pyplot as plt
fig, ax = plt.subplots(
        nrows=2, 
        ncols=1, 
        sharex=False,  # 'none', 'all', 'row', 'col'
        sharey=False,  # 'none', 'all', 'row', 'col'
        squeeze=True)

# Taylor series for Si.
def coeff(n):
    from mpmath import mp, mpf, fac
    mp.dps = 80
    return (-1) ** n / (mpf(2 * n + 1) * fac(mpf(2 * n + 1)))

# Pade approximant for Si.
a, b = pade(12, 12, coeff)
a = np.array([*reversed(a)])
b = np.array([*reversed(b)])
f = lambda x: x * np.polyval(a, x ** 2) / np.polyval(b, x ** 2)

# Compare.
x = np.linspace(-20, 20, 256)
s, c = sici(x)
ax[0].plot(x, f(x), color='r', linestyle='--')
ax[1].plot(x, s, color='r')
plt.show()
