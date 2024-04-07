import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import rc

# Use a serif font that is similar to Times New Roman
rc('font', family='serif')
rc('text', usetex=False)  # Set to True if LaTeX is installed and you want to use it

initial_mean, initial_std = 1, 1
noise_mean, noise_std = 0, 0.5
safety_threshold = 0.95
boundary_x = 1

x_range = np.linspace(-5, 5, 1000)
initial_dist = stats.norm.pdf(x_range, initial_mean, initial_std)

fixed_boundary = 1
desired_probability = 0.95

combined_std = np.sqrt(initial_std**2)
combined_dist = stats.norm.pdf(x_range, initial_mean, combined_std)

z_score_for_adjustment = stats.norm.ppf(desired_probability)
required_shift = z_score_for_adjustment * combined_std

new_mean_for_adjusted_distribution = fixed_boundary - required_shift
adjusted_dist_for_fixed_boundary = stats.norm.pdf(x_range, new_mean_for_adjusted_distribution, combined_std)

# Adjust figure size to fit IEEE column width
plt.figure(figsize=(12, 8))  # Adjust the size as needed

# plt.plot(x_range, initial_dist, label='Initial Distribution N(0, 1)', color='navy', linewidth=2)
# plt.fill_between(x_range, initial_dist, alpha=0.2, color='skyblue', label='Initial Probability Mass')

plt.plot(x_range, adjusted_dist_for_fixed_boundary, label='Adjusted Distribution for Fixed Boundary', color='green', linewidth=2)
plt.fill_between(x_range, adjusted_dist_for_fixed_boundary, alpha=0.2, color='red', label='Adjusted Probability Mass')

plt.axvline(x=fixed_boundary, color='red', linestyle='--', label=f'Fixed Boundary (x = {fixed_boundary})', linewidth=2)

# Adjust font sizes
plt.xlabel('Value', fontsize=10)
plt.ylabel('Probability Density', fontsize=10)
plt.title('Distribution for Fixed Boundary', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True)

plt.tight_layout()

# Save the figure in a high-quality format
plt.savefig('C:/Users/A490243/Desktop/Master_Thesis/Figure/adjusted_distribution.png')
plt.show()
