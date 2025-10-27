import numpy as np
import matplotlib.pyplot as plt

def field_of_view(sensor_width, focal_length):
    return 2 * np.degrees(np.arctan(sensor_width / (2 * focal_length)))

#example: Full-frame sensor (36 mm width)
sensor_width = 36
focal_lengths = [18, 35, 50, 100, 200]

fovs = [field_of_view(sensor_width, f) for f in focal_lengths]

plt.figure(figsize=(8,5))
plt.plot(focal_lengths, fovs, marker='o')
plt.title("Focal Length vs Field of View")
plt.xlabel("Focal Length (mm)")
plt.ylabel("Horizontal FOV (degrees)")
plt.gca().invert_xaxis()  # Longer focal length = narrower FOV
plt.grid(True)
plt.show()
