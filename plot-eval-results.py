import matplotlib.pyplot as plt

# File containing the data
file_path = "eval_returns_100.txt"

# Read data from the file
try:
    with open(file_path, "r") as file:
        data_file = file.readlines()

        data_sets = []
       
        for line in data_file:
            # Split the line by whitespace and convert to float
            
            values = [ float(token) for token in line.split() if token.isdigit() ]
            print(values)
            # Append the values to the data list
            data_sets.append(values)
                   
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()
except ValueError:
    print("Error: File contains non-numeric or improperly formatted data.")
    exit()

# Pad the lists in data_sets with zeros to make them the same length
max_length = max(len(data) for data in data_sets)
for data in data_sets:
    data.extend([0] * (max_length - len(data)))

# Transpose the data for plotting
data = list(zip(*data_sets))
x_values = range(1, max_length + 1)

# Plot the data
plt.figure(figsize=(10, 6))
for i, series in enumerate(data_sets):  
    plt.plot(x_values, series, marker='o', linestyle='-', label=f'Series {i + 1}')

# Add labels, title, and legend
plt.xlabel('Evaluation Step')
plt.ylabel('Return Value')
plt.title('Evaluation Returns Over Steps')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()