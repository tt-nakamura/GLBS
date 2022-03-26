import numpy as np
import matplotlib.pyplot as plt
from BinaryLens import BinaryLens

q,d = 0.789, 1.213
u0 = 0.35
phi = (133.66 + 90)/180*np.pi
x1,x2,y1,y2 = -1,1,-1,1

b = BinaryLens(q,d)

x,y = np.meshgrid(np.r_[x1:x2:128j], np.r_[y1:y2:128j])
w = x + 1j*y
m = b.mag(w)
plt.imshow(np.log(m), cmap='gray', extent=(x1,x2,y1,y2))

w = (u0 + 1j*np.linspace(-1.4, 1.4, 2))*np.exp(1j*phi)
plt.plot(np.real(w), np.imag(w), 'w', lw=0.5)

plt.axis([x1,x2,y1,y2])
plt.title('OGLE-2003BLG170')
plt.show()
