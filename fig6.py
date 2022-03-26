import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from BinaryLens import BinaryLens

q,d = 0.789, 1.213
u0,tE = 0.35, 15.2
phi = (133.66 + 90)/180*np.pi
R = 0 # 0.0027
f = 0.75 # blending ratio

t0 = 2452794.1 # time of lensing event
I0 = 18.602 # base magnitude
N = 400

b = BinaryLens(q,d)

t = np.linspace(-10,10,N) # plot range / day
w = (u0 + 1j*t/tE)*np.exp(1j*phi)
I = I0 - 2.5*np.log10((b.mag(w,R)-1)*f+1)
t_unix = 2440587.5
t = t0 + t - t_unix

plt.axis([t[0], t[-1], 18.3, 15.3])
plt.plot(t,I)

# data from:
# http://ogle.astrouw.edu.pl/ogle3/ews/2003/ews.html
t,I,Ierr = np.loadtxt('blg-170.txt').T
plt.errorbar(t - t_unix, I, Ierr, fmt='+')

plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d'))
plt.ylabel('I-band magnitude')
plt.xlabel('Universal Time (2003)')
plt.title('OGLE-2003BLG170')
plt.show()
