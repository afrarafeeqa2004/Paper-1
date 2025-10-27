import numpy as np
import matplotlib.pyplot as plt

def depth_of_field(f, N, c, d):
    H = (f**2) / (N * c) + f  #hyperfocal distance
    near = (H * d) / (H + (d - f))
    far = (H * d) / (H - (d - f)) if d < H else np.inf
    return near, far, far - near if far != np.inf else np.inf

#example: 50mm lens, f/2.8, full-frame sensor
f = 50       # focal length (mm)
N = 2.8      # aperture (f-number)
c = 0.03     # CoC for full-frame
d = 2000     # subject distance (mm) = 2 m

near, far, dof = depth_of_field(f, N, c, d)
print(f"Near focus: {near/1000:.2f} m, Far focus: {far/1000:.2f} m, DOF: {dof/1000:.2f} m")

#aperture Effect Visualization
apertures = [1.4, 2.8, 5.6, 11, 16]
dof_values = [depth_of_field(f, N, c, d)[2] for N in apertures]

plt.figure(figsize=(8,5))
plt.plot(apertures, dof_values, marker='o')
plt.title("Effect of Aperture on Depth of Field")
plt.xlabel("Aperture (f-number)")
plt.ylabel("Depth of Field (mm)")
plt.grid(True)
plt.show()
