import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

# Example data (replace this with your own data)
"""data = [
    [0.1, 0.2, 0.5, 0.8],
    [0.3, 0.6, 0.9, 0.4],
    [0.7, 0.2, 0.1, 0.5]
]

# Define custom colors for each value in the data
colors = [
    ['#FF0000', '#FFFF00', '#00FF00', '#0000FF'],
    ['#00FFFF', '#FF00FF', '#800080', '#FF4500'],
    ['#32CD32', '#8A2BE2', '#FFD700', '#008080']
]

# Create a heatmap
fig, ax = plt.subplots()
cax = ax.matshow(data, cmap=plt.cm.Blues)

# Customize the colors of each cell based on the provided color values
for i in range(len(data)):
    for j in range(len(data[0])):
        ax.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', color=colors[i][j])

# Add a colorbar
cbar = fig.colorbar(cax)

# Show the plot
plt.show()"""

import json
from scipy import stats
from scipy.stats import f_oneway
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import mixedlm
from itertools import product
from pingouin import rm_anova, pairwise_tests
import sys
import numpy

def hex_to_rgba(hex_code):
    hex_code = hex_code.lstrip('#')  # Remove '#' if present
    hex_len = len(hex_code)
    
    # Determine if the hex code has an alpha component (8 characters) or not (6 characters)
    if hex_len == 6:
        hex_code += 'FF'  # Default alpha value if not provided
    elif hex_len != 8:
        raise ValueError("Invalid hex code length. It should be either 6 or 8 characters.")

    rgba_tuple = (
        int(hex_code[0:2], 16),
        int(hex_code[2:4], 16),
        int(hex_code[4:6], 16),
        int(hex_code[6:8], 16) / 255.0  # Normalize alpha value to the range [0, 1]
    )
    
    return rgba_tuple

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

numpy.set_printoptions(threshold=sys.maxsize)
res = None
with open("resnet18_Cifar10_True_l1_all_NEW.json", "r") as results:
    res = json.load(results)["l1"]

colors = np.chararray((len(res.keys()), len(res["0"].keys())), itemsize=7)
data = np.zeros((len(res.keys()), len(res["0"].keys())))

for i, (group_idx, values_all) in enumerate(res.items()):
    for j, sparsity in enumerate(values_all.keys()):
        values = values_all[sparsity]
        del values["default"]
        del values["OT_Source_uniform_Target_importance"]

        print(values)

        accuracies = [val for val in values.values()]
        
        max_indice = accuracies.index(max(accuracies))
        print(accuracies)
        normalized_accuracies = (np.array(accuracies) - min(accuracies)+1e-8) / (max(accuracies) - min(accuracies)+1e-8)
        print(normalized_accuracies)
        normalized_accuracies = [int(255) if i == max_indice else 0 for i,acc in enumerate(normalized_accuracies)]
        print(normalized_accuracies)
        print("********")
        color = rgb_to_hex(*normalized_accuracies)
        print(color)
        colors[i][j] = color
        data[i][j] = max(accuracies)

# Create a figure and axis
# Convert color strings to RGBA tuples
colors_rgba = np.array([[to_rgba(color.decode()) for color in row] for row in colors])

# Create a heatmap with custom colors
fig, ax = plt.subplots()
cax = ax.imshow(data, cmap='viridis', interpolation='nearest')

# Customize the colors of each cell based on the provided color values
for i in range(len(data)):
    for j in range(len(data[0])):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color=colors_rgba[i][j]))

# Add a colorbar
#cbar = fig.colorbar(cax)

# Show the plot
plt.show()

def hex_to_rgba(hex_code):
    hex_code = hex_code.lstrip('#')  # Remove '#' if present
    hex_len = len(hex_code)

    if hex_len == 6:
        hex_code += 'FF'
    elif hex_len != 8:
        raise ValueError("Invalid hex code length. It should be either 6 or 8 characters.")

    rgba_tuple = (
        int(hex_code[0:2], 16),
        int(hex_code[2:4], 16),
        int(hex_code[4:6], 16),
        int(hex_code[6:8], 16) / 255.0
    )

    return rgba_tuple

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

with open("resnet18_Cifar10_True_l1_all_NEW.json", "r") as results:
    res = json.load(results)["l1"]

colors = np.chararray((len(res.keys()), len(res["0"].keys())), itemsize=7)
data = np.zeros((len(res.keys()), len(res["0"].keys())))

for i, (group_idx, values_all) in enumerate(res.items()):
    for j, sparsity in enumerate(values_all.keys()):
        values = values_all[sparsity]
        del values["default"]
        del values["OT_Source_uniform_Target_importance"]

        accuracies = [val for val in values.values()]
        normalized_accuracies = (np.array(accuracies) - min(accuracies) + 1e-8) / (
                    max(accuracies) - min(accuracies) + 1e-8)
        normalized_accuracies = [int(255 * acc) for acc in normalized_accuracies]

        color = rgb_to_hex(*normalized_accuracies)
        colors[i][j] = color
        data[i][j] = max(accuracies)

# Convert color strings to RGBA tuples
colors_rgba = np.array([[to_rgba(color.decode()) for color in row] for row in colors])

# Create a figure and axis
fig, ax = plt.subplots()

# Create a heatmap with custom colors
cax = ax.imshow(data, cmap='plasma', interpolation='nearest')

# Customize the colors of each cell based on the provided color values
for i in range(len(data)):
    for j in range(len(data[0])):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color=colors_rgba[i][j]))

# Add a colorbar
cbar = fig.colorbar(cax)

# Show the plot
plt.show()


        