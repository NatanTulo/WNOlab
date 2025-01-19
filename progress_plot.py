import matplotlib.pyplot as plt
import re

def extract_numbers_after_equal(file_path):
    """
    Extracts the last number after the '=' sign from each line in the file.
    """
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            # Use regex to find the pattern "= X.X" or "= X" where X is a number
            match = re.search(r'=\s*(-?\d+\.?\d*)', line)
            if match:
                # Extract the number and convert it to float
                number = float(match.group(1))
                numbers.append(number)
    return numbers

def plot_numbers(numbers):
    """
    Plots the extracted numbers using Matplotlib.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(numbers, marker='o', linestyle='-', color='b')
    plt.title("Kolejne podejścia szukania outlayerów")
    plt.xlabel("Numer podejścia")
    plt.ylabel("Liczba punktów")
    plt.grid(True)
    plt.show()

def main(file_path):
    """
    Main function to extract numbers and plot them.
    """
    # Extract numbers from the file
    numbers = extract_numbers_after_equal(file_path)
    
    # Plot the numbers
    plot_numbers(numbers)

if __name__ == "__main__":
    # Replace 'your_file.txt' with the path to your file
    file_path = 'outliers.txt'  # Change this to your file path
    main(file_path)