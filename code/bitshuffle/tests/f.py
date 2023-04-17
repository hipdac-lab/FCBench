from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import time

import numpy as np
from numpy import random

from bitshuffle import ext, __zstd__
K = 5
tasks = ['num_brain', 'num_control', 'msg_bt', 'hdr_night', 'h3d_temp',
         'ts_gas', 'rsim_f32', 'spitzer_irac', 'astro_mhd_f64', 'astro_pt_f64',
         'turbulence', 'wave_f32', 'citytemp_f32', 'miranda3d']
dts = ['f64', 'f64', 'f64', 'f64', 'f32',
       'f32', 'f32', 'f32', 'f64', 'f64',
       'f32', 'f32', 'f32', 'f32']
task, dt = tasks[K], dts[K]
DT = {'f32': np.float32, 'f64': np.float64}
datadir = '/home/nextnet.com/xchen/Documents/data/FP/'
path_to_X = datadir + task
X = np.fromfile(path_to_X, dtype=DT[dt])
# If we are doing timeings by what factor to increase workload.
# Remember to change `ext.REPEATC`.
TIME = 0
# TIME = 8    # 8kB blocks same as final blocking.
BLOCK = 1024


TEST_DTYPES = [
    np.uint8,
    np.uint16,
    np.int32,
    np.uint64,
    np.float32,
    np.float64,
    np.complex128,
]
TEST_DTYPES += [b"a3", b"a5", b"a6", b"a7", b"a9", b"a11", b"a12", b"a24", b"a48"]


class TestProfile(unittest.TestCase):
    def setUp(self):
        n = 1024  # bytes.
        if TIME:
            n *= TIME
        # Almost random bits, but now quite. All bits exercised (to fully test
        # transpose) but still slightly compresible.
        # self.data = random.randint(0, 200, n).astype(np.uint8)
        self.data = X
        self.fun = ext.copy
        self.check = None
        self.check_data = None
        self.case = "None"

    def tearDown(self):
        """Performs all tests and timings."""
        if TIME:
            reps = 10
        else:
            reps = 1
        delta_ts = []
        try:
            for ii in range(reps):
                t0 = time.time()
                out = self.fun(self.data)
                delta_ts.append(time.time() - t0)
        except RuntimeError as err:
            if len(err.args) > 1 and (err.args[1] == -11) and not ext.using_SSE2():
                return
            if len(err.args) > 1 and (err.args[1] == -12) and not ext.using_AVX2():
                return
            if len(err.args) > 1 and (err.args[1] == -14) and not ext.using_AVX512():
                return
            else:
                raise
        delta_t = min(delta_ts)
        size_i = self.data.size * self.data.dtype.itemsize
        size_o = out.size * out.dtype.itemsize
        size = max([size_i, size_o])
        speed = ext.REPEAT * size / delta_t / 1024**3  # GB/s
        if TIME == 0:
            print("%-20s %-20s: %5.2f s/GB,   %5.2f GB/s %d %d %4.5f" % (task, self.case, 1.0 / speed, speed, size_i, size_o, delta_ts[0]))
        if self.check is not None:
            ans = self.check(self.data).view(np.uint8)
            self.assertTrue(np.all(ans == out.view(np.uint8)))
        if self.check_data is not None:
            ans = self.check_data.view(np.uint8)
            self.assertTrue(np.all(ans == out.view(np.uint8)))

    def test_10c_compress_64(self):
        self.case = "compress 64"
        self.data = self.data.view(np.float64)
        self.fun = lambda x: ext.compress_lz4(x, BLOCK)

    def test_10d_decompress_64(self):
        self.case = "decompress 64"
        pre_trans = self.data.view(np.float64)
        self.data = ext.compress_lz4(pre_trans, BLOCK)
        self.fun = lambda x: ext.decompress_lz4(
            x, pre_trans.shape, pre_trans.dtype, BLOCK
        )
        self.check_data = pre_trans

    @unittest.skipUnless(__zstd__, "ZSTD support not included")
    def test_10c_compress_z64(self):
        self.case = "compress zstd  64"
        self.data = self.data.view(np.float64)
        self.fun = lambda x: ext.compress_zstd(x, BLOCK)

    @unittest.skipUnless(__zstd__, "ZSTD support not included")
    def test_10d_decompress_z64(self):
        self.case = "decompress zstd 64"
        pre_trans = self.data.view(np.float64)
        self.data = ext.compress_zstd(pre_trans, BLOCK)
        self.fun = lambda x: ext.decompress_zstd(
            x, pre_trans.shape, pre_trans.dtype, BLOCK
        )
        self.check_data = pre_trans


# Python implementations for checking results.


def trans_byte_elem(arr):
    dtype = arr.dtype
    itemsize = dtype.itemsize
    in_buf = arr.flat[:].view(np.uint8)
    nelem = in_buf.size // itemsize
    in_buf.shape = (nelem, itemsize)

    out_buf = np.empty((itemsize, nelem), dtype=np.uint8)
    for ii in range(nelem):
        for jj in range(itemsize):
            out_buf[jj, ii] = in_buf[ii, jj]
    return out_buf.flat[:].view(dtype)


def trans_bit_byte(arr):
    n = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    bits = np.unpackbits(arr.view(np.uint8))
    bits.shape = (n * itemsize, 8)
    # We have to reverse the order of the bits both for unpacking and packing,
    # since we want to call the least significant bit the first bit.
    bits = bits[:, ::-1]
    bits_shuff = (bits.T).copy()
    bits_shuff.shape = (n * itemsize, 8)
    bits_shuff = bits_shuff[:, ::-1]
    arr_bt = np.packbits(bits_shuff.flat[:])
    return arr_bt.view(dtype)


def trans_bit_elem(arr):
    n = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    bits = np.unpackbits(arr.view(np.uint8))
    bits.shape = (n * itemsize, 8)
    # We have to reverse the order of the bits both for unpacking and packing,
    # since we want to call the least significant bit the first bit.
    bits = bits[:, ::-1].copy()
    bits.shape = (n, itemsize * 8)
    bits_shuff = (bits.T).copy()
    bits_shuff.shape = (n * itemsize, 8)
    bits_shuff = bits_shuff[:, ::-1]
    arr_bt = np.packbits(bits_shuff.flat[:])
    return arr_bt.view(dtype)


if __name__ == "__main__":
    unittest.main()
