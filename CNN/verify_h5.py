import h5py

file_path = "/Users/bernardoquindimil/Code/Berniquindimil/Proyect/CNN/model_weights.weights.h5"

with h5py.File(file_path, "r") as f:
    print(f.keys())  # List groups in the H5 file
