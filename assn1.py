import numpy as np

def q1(n):
 # Create a base matrix with values in the upper triangle
    upper_triangle = np.triu(np.ones((n, n)), k=0)
    upper_triangle = 10 * n * upper_triangle
     # Create an array of indices from 0 to n-1
    indices = np.arange(n)
    # Compute the diagonal values using vectorized operations
    diagonal_values = 10 * indices
    # Create a matrix where the diagonal values are broadcasted across each row
    diagonal_matrix = np.diag(diagonal_values)
    # Add the upper triangle matrix to get the values in the top triangle
    result = upper_triangle - diagonal_matrix
    # Create the lower triangle values by mirroring and changing signs
    lower_triangle = np.tril(-np.ones((n, n)), k=-1)
    
    # Apply the diagonal values to the lower triangle
    lower_values = n * 10
    # Combine the upper and lower parts
    result += lower_triangle * lower_values
    
    return result

def q2(num):
    #create row and column vectors with arange to combine into a 2D array
    r = np.arange(num).reshape(-1, 1)
    c = np.arange(num).reshape(1, -1)
    arr = r + c
    return arr
def q3(num):
    values = np.arange(num) * 10
    arr = values[:, np.newaxis, np.newaxis]  # Shape: (n, 1, 1)
    # Broadcast the values to the entire (n, n, n) shape
    arr = arr * np.ones((1, num, num), dtype=int) 

    return arr
def q4(array):
     # Convert the input to a numpy array if it's not already
    array = np.asarray(array)
    
    # Calculate the magnitude for 1D array
    if array.ndim == 1:
        magnitude = np.linalg.norm(array)
        if magnitude != 0:  # Avoid division by zero
            return array / magnitude
    
    # For higher dimensions, normalize row vectors
    else:
        # Calculate the magnitudes of the row vectors
        magnitudes = np.linalg.norm(array, axis=-1, keepdims=True)
        # Avoid division by zero by checking if magnitude is zero
        magnitudes[magnitudes == 0] = 1
        return array / magnitudes

    return array
def q5(array):
    # Convert input to numpy array if it's not already
    array = np.asarray(array)
    
    # Flatten the array to make processing easier
    flat_array = array.flatten()
    
    # Identify elements that are multiples of 7 or 11
    is_multiple_of_7 = (flat_array % 7 == 0)
    is_multiple_of_11 = (flat_array % 11 == 0)
    
    # Use logical OR to find elements that are multiples of 7 or 11
    result = flat_array[is_multiple_of_7 | is_multiple_of_11]
    
    # Compute the sum of these elements
    return np.sum(result)

def q6(num):
    # Create arrays for row and column indices
    rows = np.arange(1, num + 1).reshape(-1, 1)  # Shape (n, 1)
    cols = np.arange(num)  # Shape (n,)
    
    # Compute the power values using broadcasting
    result = rows ** cols
    
    return result


def q7(arr, k):
    # Validate input
    if k < 1:
        raise ValueError("k must be >= 1")
    
    # Determine the new shape
    shape = np.array(arr.shape)
    shape[-2:] += 2 * k
    
    # Create a new array filled with zeros of the new shape
    padded_array = np.zeros(shape, dtype=arr.dtype)
    
    # Calculate slices for inserting the original array
    slices = [slice(None)] * arr.ndim
    slices[-2] = slice(k, k + arr.shape[-2])
    slices[-1] = slice(k, k + arr.shape[-1])
    
    # Insert the original array into the padded array
    padded_array[tuple(slices)] = arr
    
    return padded_array

def main():
  print(q7(np.array([[1, 2], [3, 4]]), 2))
    

if __name__ == "__main__":
    main()