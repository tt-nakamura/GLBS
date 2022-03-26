import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from matplotlib.dates import DateFormatter

# amplification factor for point mass lens
def Ampl(u): return (u*u+2)/u/np.sqrt(u*u+4)
def Ampl_p(u): return -8/u/u/(u*u + 4)**1.5

t0 = 2456081.949 # time of lensing event (Julius date)
tE = 17.427 # time to traverse Einstein radius / day
A0 = 2.627 # max amplification
I0 = 16.867 # base magnitude

N = 200
t = np.linspace(-62, 62, N) # plot range / day

# closest separation between lens and source
u0 = newton(lambda u: Ampl(u) - A0, 0.1, Ampl_p)

u = np.sqrt((t/tE)**2 + u0**2)
I = I0 - 2.5*np.log10(Ampl(u))
t_unix = 2440587.5
t = t0 + t - t_unix

plt.axis([t[0], t[-1], 16.9, 15.7])
plt.plot(t,I)

# data from:
# http://ogle.astrouw.edu.pl/ogle4/ews/2012/ews.html
t,I,Ierr,_,_ = np.loadtxt('blg-0631.txt').T
plt.errorbar(t - t_unix, I, Ierr, fmt='+')

plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d'))
plt.ylabel('I-band magnitude')
plt.xlabel('Universal Time (2012)')
plt.title('OGLE-2012BLG631')
plt.show()
