import h5py
import numpy
import bitshuffle.h5

print(h5py.__version__) # >= '2.5.0'
filename = '/tmp/h5data'
f = h5py.File(filename, "w")

# block_size = 0 let Bitshuffle choose its value
block_size = 0

dataset = f.create_dataset(
    "data",
    (100, 100, 100),
    compression=bitshuffle.h5.H5FILTER,
    compression_opts=(block_size, bitshuffle.h5.H5_COMPRESS_ZSTD),
    dtype='float32',
    )

# create some random data
array = numpy.random.rand(100, 100, 100)
array = array.astype('float32')

dataset[:] = array

f.close()