import h5py
import pickle

# Load matrices on hdf5 format and store in pickle format

files = ["mycielskian10", "mycielskian10", "mycielskian11", "Goodwin_010", "Goodwin_013"]

for file_n in files:
    file_name = file_n + '.mat'
    f = h5py.File(file_name, 'r')

    # Extract relevant data
    group = f['Problem']
    data = group['A']['data'][()]
    ir = group['A']['ir'][()]
    jc = group['A']['jc'][()]

    matrix = [data, ir, jc]

    with open(file_n + '.pickle', 'wb') as file:
        pickle.dump(matrix, file, protocol=2)

    with open(file_n + '.pickle', 'rb') as file:
        matrix = pickle.load(file)