import numpy as np
import matplotlib.pyplot as plt
from BinaryLens import BinaryLens

q,d = 0.5, 1
w0,r1,r2 = 0.1, 0.1, 0.05
N = 400
w1 = w0 + r1*np.exp(1j*np.linspace(0, 2*np.pi, N))
w2 = w0 + r2*np.exp(1j*np.linspace(0, 2*np.pi, N))

b = BinaryLens(q,d)

zc = b.crit()
wc = b.map(zc)
z1 = b.image(w1)
z2 = b.image(w2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(wc), np.imag(wc), 'k:')
plt.plot(np.real(w1), np.imag(w1), 'r')
plt.plot(np.real(w2), np.imag(w2), 'b')
plt.title('source plane')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(zc), np.imag(zc), 'k:')
plt.plot(np.real(z1), np.imag(z1), 'r')
plt.plot(np.real(z2), np.imag(z2), 'b')
plt.plot(np.real(b.z), np.imag(b.z), '+')
plt.title('lens plane')

plt.show()
#######################################################

q,d = 0.1, 1
w0,r1,r2 = 0.1, 0.1, 0.05
N = 400
w1 = w0 + r1*np.exp(1j*np.linspace(0, 2*np.pi, N))
w2 = w0 + r2*np.exp(1j*np.linspace(0, 2*np.pi, N))

b = BinaryLens(q,d)

zc = b.crit()
wc = b.map(zc)
z1 = b.image(w1)
z2 = b.image(w2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(wc), np.imag(wc), 'k:')
plt.plot(np.real(w1), np.imag(w1), 'r')
plt.plot(np.real(w2), np.imag(w2), 'b')
plt.title('source plane')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(zc), np.imag(zc), 'k:')
plt.plot(np.real(z1), np.imag(z1), 'r')
plt.plot(np.real(z2), np.imag(z2), 'b')
plt.plot(np.real(b.z), np.imag(b.z), '+')
plt.title('lens plane')

plt.show()
#######################################################

q,d = 0.01, 1
w0,r1,r2 = 0.1, 0.1, 0.05
N = 400
w1 = w0 + r1*np.exp(1j*np.linspace(0, 2*np.pi, N))
w2 = w0 + r2*np.exp(1j*np.linspace(0, 2*np.pi, N))

b = BinaryLens(q,d)

zc = b.crit()
wc = b.map(zc)
z1 = b.image(w1)
z2 = b.image(w2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(wc), np.imag(wc), 'k:')
plt.plot(np.real(w1), np.imag(w1), 'r')
plt.plot(np.real(w2), np.imag(w2), 'b')
plt.title('source plane')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(zc), np.imag(zc), 'k:')
plt.plot(np.real(z1), np.imag(z1), 'r')
plt.plot(np.real(z2), np.imag(z2), 'b')
plt.plot(np.real(b.z), np.imag(b.z), '+')
plt.title('lens plane')

plt.show()
#######################################################

q,d = 0.5, 2
w0,r1,r2 = -0.3, 0.1, 0.05
N = 400
w1 = w0 + r1*np.exp(1j*np.linspace(0, 2*np.pi, N))
w2 = w0 + r2*np.exp(1j*np.linspace(0, 2*np.pi, N))

b = BinaryLens(q,d)

zc = b.crit()
wc = b.map(zc)
z1 = b.image(w1)
z2 = b.image(w2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(wc), np.imag(wc), 'k:')
plt.plot(np.real(w1), np.imag(w1), 'r')
plt.plot(np.real(w2), np.imag(w2), 'b')
plt.title('source plane')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(zc), np.imag(zc), 'k:')
plt.plot(np.real(z1), np.imag(z1), 'r')
plt.plot(np.real(z2), np.imag(z2), 'b')
plt.plot(np.real(b.z), np.imag(b.z), '+')
plt.title('lens plane')

plt.show()
#######################################################

q,d = 0.5, 0.8
w0,r1,r2 = 0.1, 0.1, 0.05
N = 400
w1 = w0 + r1*np.exp(1j*np.linspace(0, 2*np.pi, N))
w2 = w0 + r2*np.exp(1j*np.linspace(0, 2*np.pi, N))

b = BinaryLens(q,d)

zc = b.crit()
wc = b.map(zc)
z1 = b.image(w1)
z2 = b.image(w2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(wc), np.imag(wc), 'k:')
plt.plot(np.real(w1), np.imag(w1), 'r')
plt.plot(np.real(w2), np.imag(w2), 'b')
plt.title('source plane')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(zc), np.imag(zc), 'k:')
plt.plot(np.real(z1), np.imag(z1), 'r')
plt.plot(np.real(z2), np.imag(z2), 'b')
plt.plot(np.real(b.z), np.imag(b.z), '+')
plt.title('lens plane')

plt.show()
#######################################################

q,d = 0.5, 0.7
w0,r1,r2 = 0.1, 0.1, 0.05
N = 400
w1 = w0 + r1*np.exp(1j*np.linspace(0, 2*np.pi, N))
w2 = w0 + r2*np.exp(1j*np.linspace(0, 2*np.pi, N))

b = BinaryLens(q,d)

zc = b.crit()
wc = b.map(zc)
z1 = b.image(w1)
z2 = b.image(w2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(wc), np.imag(wc), 'k:')
plt.plot(np.real(w1), np.imag(w1), 'r')
plt.plot(np.real(w2), np.imag(w2), 'b')
plt.title('source plane')

plt.subplot(1,2,2)
plt.axis('equal')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.plot(np.real(zc), np.imag(zc), 'k:')
plt.plot(np.real(z1), np.imag(z1), 'r')
plt.plot(np.real(z2), np.imag(z2), 'b')
plt.plot(np.real(b.z), np.imag(b.z), '+')
plt.title('lens plane')

plt.show()
