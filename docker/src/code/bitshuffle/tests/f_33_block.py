from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import unittest
import time
import os
import numpy as np
from numpy import random

from bitshuffle import ext, __zstd__
K = 0
tasks = ['msg_bt_f64',   
         'num_brain_f64',   
         'num_control_f64',   
         'rsim_f32',   
         'astro_mhd_f64',   
         'astro_pt_f64',   
         # 'miranda3d_f32',   
         'turbulence_f32',   
         'wave_f32',   
         'h3d_temp_f32',   
         'citytemp_f32',   
         'ts_gas_f32',   
         'phone_gyro_f64',   
         'wesad_chest_f64',   
         'jane_street_f64',   
         'nyc_taxi2015_f64',   
         'spain_gas_price_f64',   
         'solar_wind_f32',   
         'acs_wht_f32',   
         'hdr_night_f32',   
         'hdr_palermo_f32',   
         'hst_wfc3_uvis_f32',   
         'hst_wfc3_ir_f32',   
         'spitzer_irac_f32',   
         'g24_78_usb2_f32',   
         'jw_mirimage_f32',   
         'tpch_order_f64',   
         'tpcxbb_store_f64',   
         'tpcxbb_web_f64',   
         'tpch_lineitem_f32',   
         'tpcds_catalog_f32',   
         'tpcds_store_f32']   
         # 'tpcds_web_f32']   
dts = ['f64',   
       'f64',   
       'f64',   
       'f32',   
       'f64',   
       'f64',   
       # 'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f64',   
       'f64',   
       'f64',   
       'f64',   
       'f64',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f32',   
       'f64',   
       'f64',   
       'f64',   
       'f32',   
       'f32',   
       'f32']   
       # 'f32']
bks = [512,   
       512,   
       512,   
       1024,   
       512,   
       512,   
       # 'f32',   
       1024,   
       1024,   
       1024,   
       1024,   
       1024,   
       512,   
       512,   
       512,   
       512,   
       512,   
       1024,   
       1024,   
       1024,   
       1024,   
       1024,   
       1024,   
       1024,   
       1024,   
       1024,   
       512,   
       512,   
       512,   
       1024,   
       1024,   
       1024]   
       # 'f32']       
# If we are doing timeings by what factor to increase workload.
# Remember to change `ext.REPEATC`.
TIME = 0
# TIME = 8    # 8kB blocks same as final blocking.
BLOCK=1024


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
            # print("%-20s %-20s: %5.2f s/GB,   %5.2f GB/s %d %d %4.5f" % (task, self.case, 1.0 / speed, speed, size_i, size_o, delta_ts[0]))
            print("%-20s %-20s: %5.2f GB/s %d %d %4.5f" % (task, self.case, speed, size_i, size_o, size_i/size_o))
        if self.check is not None:
            ans = self.check(self.data).view(np.uint8)
            self.assertTrue(np.all(ans == out.view(np.uint8)))
        if self.check_data is not None:
            ans = self.check_data.view(np.uint8)
            self.assertTrue(np.all(ans == out.view(np.uint8)))
        try:
            with open("%s.btsf"%(task), 'wb') as bf:
                bf.write(out.view(np.uint8))
        except Exception as e:
            print(e)    

    def test_10c_compress_64(self):
        self.case = "compress_LZ_64"
        self.data = self.data.view(np.float64)
        self.fun = lambda x: ext.compress_lz4(x, BLOCK)

    def test_10d_decompress_64(self):
        self.case = "decompress_LZ_64"
        pre_trans = self.data.view(np.float64)
        self.data = ext.compress_lz4(pre_trans, BLOCK)
        self.fun = lambda x: ext.decompress_lz4(
            x, pre_trans.shape, pre_trans.dtype, BLOCK
        )
        self.check_data = pre_trans

    @unittest.skipUnless(__zstd__, "ZSTD support not included")
    def test_10c_compress_z64(self):
        self.case = "compress_zstd_64"
        self.data = self.data.view(np.float64)
        self.fun = lambda x: ext.compress_zstd(x, BLOCK)

    @unittest.skipUnless(__zstd__, "ZSTD support not included")
    def test_10d_decompress_z64(self):
        self.case = "decompress_zstd_64"
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
    if len(sys.argv) > 1:
        K_ = sys.argv.pop()
        K = int(K_)    
    if len(sys.argv) > 1:        
        FACTOR_ = sys.argv.pop()
        FACTOR = int(FACTOR_)
        BLOCK = FACTOR * bks[K]
    print("argc=%d, K=%d, BLOCK=%d" % (len(sys.argv), K, BLOCK))
    task, dt = tasks[K], dts[K]
    DT = {'f32': np.float32, 'f64': np.float64}
    key = 'WORKDIR'
    datadir = os.path.join(os.getenv(key),'data')
    path_to_X = os.path.join(datadir, task)
    X = np.fromfile(path_to_X, dtype=DT[dt])
    print("X.shape=",X.shape)
    unittest.main()
