import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from BinaryLens import BinaryLens

q,d = 0.0067, 0.758
u0,tE = 0.0225, 73.9
phi = (274.48 - 90)/180*np.pi
R = 0 # 0.001
N = 400

t0 = 2453480.6919 # time of lensing event
Is,Ib = 19.59, 21.05
f = 1/(1 + 10**(0.4*(Is-Ib))) # blending ratio
I0 = 19.11 # base magnitude

b = BinaryLens(q,d)

t = np.linspace(-10,10,N) # plot range / day
w = (u0 + 1j*t/tE)*np.exp(1j*phi)
I = I0 - 2.5*np.log10((b.mag(w,R)-1)*f+1)
t_unix = 2440587.5
t = t0 + t - t_unix

plt.axis([t[0], t[-1], 17.3, 14.8])
plt.plot(t,I)

# data from:
# http://ogle.astrouw.edu.pl/ogle3/ews/2005/ews.html
t,I,Ierr = np.loadtxt('blg-071.txt').T
plt.errorbar(t - t_unix, I, Ierr, fmt='+')

plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d'))
plt.ylabel('I-band magnitude')
plt.xlabel('Universal Time (2005)')
plt.title('OGLE-2005BLG071')
plt.show()
