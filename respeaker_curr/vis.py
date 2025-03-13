import numpy as np
import matplotlib.pyplot as plt

def calculate_trapezoid(person_pos, array_center, front_width=0.6, back_width=1.0, height=1.0):
    # Vector from array center to person position
    direction = np.array(person_pos) - np.array(array_center)
    angle = np.arctan2(direction[1], direction[0])

    # Rotation matrix to align with direction
    rotation_matrix = np.array([
        [-np.cos(angle), -np.sin(angle)],
        [np.sin(angle), -np.cos(angle)]
    ])

    # Define trapezoid corners relative to the person position
    trapezoid_local = np.array([
        [-back_width / 2, -height / 2],   # Bottom-left (further side)
        [back_width / 2, -height / 2],    # Bottom-right (further side)
        [front_width / 2, height / 2],    # Top-right (closer side)
        [-front_width / 2, height / 2]    # Top-left (closer side)
    ])

    # Rotate and translate to global position
    trapezoid_global = np.dot(trapezoid_local, rotation_matrix.T) + person_pos

    return trapezoid_global

# Visualization
array_center = np.array([0, 0])
person_pos = np.array([1, 1])
trapezoid = calculate_trapezoid(person_pos, array_center)

plt.plot([p[0] for p in np.vstack((trapezoid, trapezoid[0]))], 
         [p[1] for p in np.vstack((trapezoid, trapezoid[0]))], 'b-')

plt.scatter(*array_center, color='r', label='Array Center')
plt.scatter(*person_pos, color='g', label='Person Position')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.axis('equal')
plt.legend()
plt.show()