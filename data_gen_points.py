import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUM_SAMPLES = 10
NUM_LEVELS = 12
NUM_MEASUREMENTS = 11

# Define dimensions and resolution
height_m, width_m = 1.5, 1.5
height_px, width_px = 300, 300
fading_distance = 0.15  # meters

# Convert fading distance and light source heights to pixels
fading_distance_px = fading_distance / height_m * height_px

# Generate the zigzag path

y_positions = np.linspace(0, height_px, NUM_LEVELS + 1).astype(int)
x_positions = np.zeros_like(y_positions)
x_positions[1::2] = width_px - 1
x_positions[0::2] = 0

# Define measurement points
measurement_y_positions = np.linspace(0, height_px - 1, NUM_MEASUREMENTS)

# Measurement points at center, left center, and right center
center_x = width_px // 2
left_x = width_px // 4
right_x = 3 * width_px // 4

measurement_x_positions_center = center_x * np.ones_like(
    measurement_y_positions, dtype=int
)
measurement_x_positions_left = left_x * np.ones_like(measurement_y_positions, dtype=int)
measurement_x_positions_right = right_x * np.ones_like(
    measurement_y_positions, dtype=int
)

# Convert to meters for plotting
measurement_x_positions_center_m = measurement_x_positions_center / width_px * width_m
measurement_x_positions_left_m = measurement_x_positions_left / width_px * width_m
measurement_x_positions_right_m = measurement_x_positions_right / width_px * width_m
measurement_y_positions_m = height_m - measurement_y_positions / height_px * height_m


# Function to calculate gradient image and measurements
def calculate_measurements(intensity_light_sources):
    # Create the gradient image
    gradient_image = np.zeros((height_px, width_px))

    light_source1_y = int(0.4 / height_m * height_px)
    light_source2_y = int(1.0 / height_m * height_px)
    light_source1_y = np.full(height_px, light_source1_y)
    light_source2_y = np.full(height_px, light_source2_y)

    # Calculate distances and intensities
    y_indices = np.arange(height_px)
    distance1_y = np.abs(y_indices[:, None] - light_source1_y[None, :])
    distance2_y = np.abs(y_indices[:, None] - light_source2_y[None, :])

    intensity1_y = intensity_light_sources * np.exp(
        -0.5 * (distance1_y / fading_distance_px) ** 2
    )
    intensity2_y = intensity_light_sources * np.exp(
        -0.5 * (distance2_y / fading_distance_px) ** 2
    )

    gradient_image = np.clip(intensity1_y + intensity2_y, 0, 1)

    # Convert measurement positions from meters to pixels
    measurement_x_positions_center_px = (
        measurement_x_positions_center_m / width_m
    ) * width_px
    measurement_x_positions_left_px = (
        measurement_x_positions_left_m / width_m
    ) * width_px
    measurement_x_positions_right_px = (
        measurement_x_positions_right_m / width_m
    ) * width_px
    measurement_y_positions_px = (
        (height_m - measurement_y_positions_m) / height_m * height_px
    )

    left = [int(x) for x in measurement_x_positions_left_px][0]
    middle = [int(x) for x in measurement_x_positions_center_px][0]
    right = [int(x) for x in measurement_x_positions_right_px][0]

    samples = []
    for a in [left, middle, right]:
        for y in measurement_y_positions_px:
            b = int(y)
            samples.append(gradient_image[a][b])

    return gradient_image, np.array(samples)


# Light source intensities to test
intensities = np.linspace(0.1, 2.0, NUM_SAMPLES)


# Create dataset
data = []

for intensity in intensities:
    gradient_image, measurements = calculate_measurements(intensity)
    avg_measurement = np.mean(measurements)
    measurements = np.round(measurements, 4)
    data.append(
        [np.round(intensity, 4)] + list(measurements) + [np.round(avg_measurement, 2)]
    )

# Convert data to DataFrame and save to CSV
columns = ["intensity"] + [f"m_{i+1}" for i in range(3 * NUM_MEASUREMENTS)] + ["m_avg"]
df = pd.DataFrame(data, columns=columns)

# Create the 'datasets' folder if it does not exist
if not os.path.exists("datasets"):
    os.makedirs("datasets")

# Save the DataFrame to a CSV file
csv_path = os.path.join("datasets", "light_source_measurements.csv")
df.to_csv(csv_path, index=False)

print(f"Dataset saved to {csv_path}")


# Plot and save image for the current intensity
plt.figure(figsize=(10, 6))
plt.imshow(
    gradient_image, cmap="gray", origin="upper", extent=[0, width_m, 0, height_m]
)
plt.colorbar(label="Intensity")
plt.title(
    f"Vertical Gradient with Zigzag Line and Measurement Points (Intensity = {intensity})"
)
plt.xlabel("Width (meters)")
plt.ylabel("Height (meters)")

# Plot zigzag line
plt.plot(
    x_positions / width_px * width_m,
    height_m - y_positions / height_px * height_m,
    "r-",
    linewidth=2,
)

# Plot measurement points
plt.scatter(
    measurement_x_positions_center_m,
    measurement_y_positions_m,
    color="blue",
    marker="o",
    s=50,
    label="Center Measurement Points",
)

plt.scatter(
    measurement_x_positions_left_m,
    measurement_y_positions_m,
    color="green",
    marker="o",
    s=50,
    label="Left Center Measurement Points",
)

plt.scatter(
    measurement_x_positions_right_m,
    measurement_y_positions_m,
    color="red",
    marker="o",
    s=50,
    label="Right Center Measurement Points",
)

plt.grid(True)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35))

# Create the 'images' folder if it does not exist
if not os.path.exists("images"):
    os.makedirs("images")

# Save the figure to the 'images' folder
image_path = os.path.join(
    "images", f"gradient_with_zigzag_line_and_measurements_{intensity}.png"
)
plt.savefig(image_path, bbox_inches="tight")
plt.close()  # Close the figure to free up memory

print(f"Image saved to {image_path}")
