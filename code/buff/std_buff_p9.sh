#!/bin/bash
# ============== #
# test MPC #
# ============== #
EXEDIR=/home/cc/code/buff/database
DATADIR=/home/cc/data/
RESDIR=/home/cc/experiments
# ================ #
# test MPC #
# ================ #
# cargo +nightly build --release  --package buff --bin comp_profiler
cd $EXEDIR
$EXEDIR/target/release/comp_profiler $DATADIR/msg_bt_f64          buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/num_brain_f64       buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/num_control_f64     buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/rsim_f32            buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/astro_mhd_f64       buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/astro_pt_f64        buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/miranda3d_f32       buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/turbulence_f32      buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/wave_f32            buff-simd32 100000000 1.15
# $EXEDIR/target/release/comp_profiler $DATADIR/h3d_temp_f32        buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/citytemp_f32        buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/ts_gas_f32          buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/phone_gyro_f64      buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/wesad_chest_f64     buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/jane_street_f64     buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/nyc_taxi2015_f64    buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/spain_gas_price_f64 buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/solar_wind_f32      buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/acs_wht_f32         buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/hdr_night_f32       buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/hdr_palermo_f32     buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/hst_wfc3_uvis_f32   buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/hst_wfc3_ir_f32     buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/spitzer_irac_f32    buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/g24_78_usb2_f32     buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/jw_mirimage_f32     buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpch_order_f64      buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpcxbb_store_f64    buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpcxbb_web_f64      buff-simd64 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpch_lineitem_f32   buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpcds_catalog_f32   buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpcds_store_f32     buff-simd32 100000000 1.15
$EXEDIR/target/release/comp_profiler $DATADIR/tpcds_web_f32       buff-simd32 100000000 1.15
