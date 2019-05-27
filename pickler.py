import h5py
import pickle

files = ["mycielskian10", "mycielskian10"]

for file_n in files:
    file_name = file_n + '.mat'
    f = h5py.File(file_name, 'r')

    group = f['Problem']
    data = group['A']['data'][()]
    ir = group['A']['ir'][()]
    jc = group['A']['jc'][()]

    matrix = [data, ir, jc]

    with open(file_n + '.pickle', 'wb') as file:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(matrix, file, protocol=2)#pickle.HIGHEST_PROTOCOL)


    #with open('gs://dataproc-df0b072e-030f-4a8d-9eae-4572ba8d996c-europe-north1/mycielskian3.pickle', 'rb') as file:
    with open(file_n + '.pickle', 'rb') as file:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        matrix = pickle.load(file)