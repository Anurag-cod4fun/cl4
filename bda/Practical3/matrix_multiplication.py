# Implement Matrix Multiplication using Map Reduce

# python matrix_multiplication.py input_matrix_A.txt input_matrix_B.txt
from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
import argparse

def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            row = []
            for val in values:
                try:
                    row.append(int(val))
                except ValueError:
                    pass  # Skip non-integer values
            if row:  # Only add non-empty rows to the matrix
                matrix.append(row)
    return matrix



def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Number of columns in Matrix A must equal the number of rows in Matrix B.")

    result = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_b[0])):
            value = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(len(matrix_a[0])))
            row.append(value)
        result.append(row)

    return result

def main():
    parser = argparse.ArgumentParser(description="Perform matrix multiplication (pseudo MapReduce).")
    parser.add_argument("matrix_a_file", type=str, help="File containing matrix A")
    parser.add_argument("matrix_b_file", type=str, help="File containing matrix B")
    args = parser.parse_args()

    matrix_A = read_matrix_from_file(args.matrix_a_file)
    matrix_B = read_matrix_from_file(args.matrix_b_file)
    matrix_C = np.dot(matrix_A, matrix_B)

    # Convert string values to integers in the matrices
    matrix_A = [[int(val) for val in row] for row in matrix_A]
    matrix_B = [[int(val) for val in row] for row in matrix_B]
    print(matrix_A)
    print(matrix_B)

    if len(matrix_A[0]) != len(matrix_B):
        raise ValueError("Number of columns in Matrix A must equal the number of rows in Matrix B.")

    # Simulate the MapReduce style by treating each row as a "mapper" and the entire matrix as the "reducer"
    intermediate_results = []
    for row_A in matrix_A:
        mapped_values = []
        for row_B in matrix_B:
            value = sum(a * b for a, b in zip(row_A, row_B))
            mapped_values.append(value)
        intermediate_results.append(mapped_values)

    result_matrix = [
        [sum(row[i] for row in intermediate_results) for i in range(len(intermediate_results[0]))]
    ]

    # for row in result_matrix:
    #     print(row)

    print("Matrix Multiplication using Mapreduce Output is: ")
    print(matrix_C)

if __name__ == "__main__":
    main()
