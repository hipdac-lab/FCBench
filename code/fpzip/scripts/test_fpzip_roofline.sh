#!/bin/bash
# ============== #
# test pFPC #
# ============== #
DATADIR=/home/cc/data
OUTDIR=/home/cc/output
RESDIR=/home/cc/experiments
export ADVIXE_EXPERIMENTAL=roofline
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_msg_bt_f64          -- ../build/bin/fpzip -i $DATADIR/msg_bt_f64           -t double  -1  33298679       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_num_brain_f64       -- ../build/bin/fpzip -i $DATADIR/num_brain_f64        -t double  -1  17730000       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_num_control_f64     -- ../build/bin/fpzip -i $DATADIR/num_control_f64      -t double  -1  19938093       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_rsim_f32            -- ../build/bin/fpzip -i $DATADIR/rsim_f32             -t float   -2  2048 11509     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_astro_mhd_f64       -- ../build/bin/fpzip -i $DATADIR/astro_mhd_f64        -t double  -3  130 514 1026   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_astro_pt_f64        -- ../build/bin/fpzip -i $DATADIR/astro_pt_f64         -t double  -3  512 256 640    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_miranda3d_f32       -- ../build/bin/fpzip -i $DATADIR/miranda3d_f32        -t float   -3  1024 1024 1024 
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_turbulence_f32      -- ../build/bin/fpzip -i $DATADIR/turbulence_f32       -t float   -3  256 256 256    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_wave_f32            -- ../build/bin/fpzip -i $DATADIR/wave_f32             -t float   -3  512 512 512    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_h3d_temp_f32        -- ../build/bin/fpzip -i $DATADIR/h3d_temp_f32         -t float   -3  100 500 500    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_citytemp_f32        -- ../build/bin/fpzip -i $DATADIR/citytemp_f32         -t float   -1  2906326        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_ts_gas_f32          -- ../build/bin/fpzip -i $DATADIR/ts_gas_f32           -t float   -1  76863200       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_phone_gyro_f64      -- ../build/bin/fpzip -i $DATADIR/phone_gyro_f64       -t double  -2  13932632 3     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_wesad_chest_f64     -- ../build/bin/fpzip -i $DATADIR/wesad_chest_f64      -t double  -2  4255300 8      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_jane_street_f64     -- ../build/bin/fpzip -i $DATADIR/jane_street_f64      -t double  -2  1664520 136    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_nyc_taxi2015_f64    -- ../build/bin/fpzip -i $DATADIR/nyc_taxi2015_f64     -t double  -2  12748986 7     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_spain_gas_price_f64 -- ../build/bin/fpzip -i $DATADIR/spain_gas_price_f64  -t double  -2  36942486 3     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_solar_wind_f32      -- ../build/bin/fpzip -i $DATADIR/solar_wind_f32       -t float   -2  7571081 14     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_acs_wht_f32         -- ../build/bin/fpzip -i $DATADIR/acs_wht_f32          -t float   -2  7500 7500      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_hdr_night_f32       -- ../build/bin/fpzip -i $DATADIR/hdr_night_f32        -t float   -2  8192 16384     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_hdr_palermo_f32     -- ../build/bin/fpzip -i $DATADIR/hdr_palermo_f32      -t float   -2  10268 20536    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_hst_wfc3_uvis_f32   -- ../build/bin/fpzip -i $DATADIR/hst_wfc3_uvis_f32    -t float   -2  5329 5110      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_hst_wfc3_ir_f32     -- ../build/bin/fpzip -i $DATADIR/hst_wfc3_ir_f32      -t float   -2  2484 2417      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_spitzer_irac_f32    -- ../build/bin/fpzip -i $DATADIR/spitzer_irac_f32     -t float   -2  6456 6389      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_g24_78_usb2_f32     -- ../build/bin/fpzip -i $DATADIR/g24_78_usb2_f32      -t float   -3  2426 371 371   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_jw_mirimage_f32     -- ../build/bin/fpzip -i $DATADIR/jw_mirimage_f32      -t float   -3  40 1024 1032   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpch_order_f64      -- ../build/bin/fpzip -i $DATADIR/tpch_order_f64       -t double  -1  15000000       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpcxbb_store_f64    -- ../build/bin/fpzip -i $DATADIR/tpcxbb_store_f64     -t double  -2  8228343 12     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpcxbb_web_f64      -- ../build/bin/fpzip -i $DATADIR/tpcxbb_web_f64       -t double  -2  8223189 15     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpch_lineitem_f32   -- ../build/bin/fpzip -i $DATADIR/tpch_lineitem_f32    -t float   -2  59986051 4     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpcds_catalog_f32   -- ../build/bin/fpzip -i $DATADIR/tpcds_catalog_f32    -t float   -2  2880058 15     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpcds_store_f32     -- ../build/bin/fpzip -i $DATADIR/tpcds_store_f32      -t float   -2  5760749 12     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/fpzip_tpcds_web_f32       -- ../build/bin/fpzip -i $DATADIR/tpcds_web_f32        -t float   -2  1439247 15     
