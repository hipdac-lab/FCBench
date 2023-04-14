# ============== #
# test nvcomp #
# ============== #
EXECDIR=/home/cc/code/nvbench/bin
DATADIR=/home/cc/data
RESDIR=/home/cc/experiments
# ================ #
# test MPC #
# ================ #
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/msg_bt_f64           >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/num_brain_f64        >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/num_control_f64      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/rsim_f32             >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/astro_mhd_f64        >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/astro_pt_f64         >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/miranda3d_f32        >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/turbulence_f32       >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/wave_f32             >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/h3d_temp_f32         >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/citytemp_f32         >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/ts_gas_f32           >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/phone_gyro_f64       >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/wesad_chest_f64      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/jane_street_f64      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/nyc_taxi2015_f64     >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/spain_gas_price_f64  >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/solar_wind_f32       >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/acs_wht_f32          >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/hdr_night_f32        >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/hdr_palermo_f32      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/hst_wfc3_uvis_f32    >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/hst_wfc3_ir_f32      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/spitzer_irac_f32     >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/g24_78_usb2_f32      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/jw_mirimage_f32      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpch_order_f64       >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcxbb_store_f64     >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcxbb_web_f64       >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpch_lineitem_f32    >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcds_catalog_f32    >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcds_store_f32      >> $RESDIR/nvlz4.txt
$EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcds_web_f32        >> $RESDIR/nvlz4.txt
# ================
# test decompress
# ================
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/msg_bt_f64           >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/num_brain_f64        >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/num_control_f64      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/rsim_f32             >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/astro_mhd_f64        >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/astro_pt_f64         >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/miranda3d_f32        >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/turbulence_f32       >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/wave_f32             >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/h3d_temp_f32         >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/citytemp_f32         >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/ts_gas_f32           >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/phone_gyro_f64       >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/wesad_chest_f64      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/jane_street_f64      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/nyc_taxi2015_f64     >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/spain_gas_price_f64  >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/solar_wind_f32       >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/acs_wht_f32          >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hdr_night_f32        >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hdr_palermo_f32      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hst_wfc3_uvis_f32    >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hst_wfc3_ir_f32      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/spitzer_irac_f32     >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/g24_78_usb2_f32      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/jw_mirimage_f32      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpch_order_f64       >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcxbb_store_f64     >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcxbb_web_f64       >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpch_lineitem_f32    >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcds_catalog_f32    >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcds_store_f32      >> $RESDIR/nvbitcomp.txt
$EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcds_web_f32        >> $RESDIR/nvbitcomp.txt
