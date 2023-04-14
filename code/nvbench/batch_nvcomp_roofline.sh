# ============== #
# test nvcomp #
# ============== #
EXECDIR=/home/cc/code/nvbench/bin
DATADIR=/home/cc/data
RESDIR=/home/cc/experiments
NCRES=/home/cc/ncres
suncu='sudo /usr/local/cuda-11.2/bin/ncu'
# ================ #
# test MPC #
# ================ #
$suncu -o $NCRES/nvlz4_msg_bt_f64           -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/msg_bt_f64           
$suncu -o $NCRES/nvlz4_num_brain_f64        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/num_brain_f64        
$suncu -o $NCRES/nvlz4_num_control_f64      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/num_control_f64      
$suncu -o $NCRES/nvlz4_rsim_f32             -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/rsim_f32             
$suncu -o $NCRES/nvlz4_astro_mhd_f64        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/astro_mhd_f64        
$suncu -o $NCRES/nvlz4_astro_pt_f64         -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/astro_pt_f64         
$suncu -o $NCRES/nvlz4_miranda3d_f32        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/miranda3d_f32        
$suncu -o $NCRES/nvlz4_turbulence_f32       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/turbulence_f32       
$suncu -o $NCRES/nvlz4_wave_f32             -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/wave_f32             
$suncu -o $NCRES/nvlz4_h3d_temp_f32         -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/h3d_temp_f32         
$suncu -o $NCRES/nvlz4_citytemp_f32         -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/citytemp_f32         
$suncu -o $NCRES/nvlz4_ts_gas_f32           -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/ts_gas_f32           
$suncu -o $NCRES/nvlz4_phone_gyro_f64       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/phone_gyro_f64       
$suncu -o $NCRES/nvlz4_wesad_chest_f64      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/wesad_chest_f64      
$suncu -o $NCRES/nvlz4_jane_street_f64      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/jane_street_f64      
$suncu -o $NCRES/nvlz4_nyc_taxi2015_f64     -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/nyc_taxi2015_f64     
$suncu -o $NCRES/nvlz4_spain_gas_price_f64  -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/spain_gas_price_f64  
$suncu -o $NCRES/nvlz4_solar_wind_f32       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/solar_wind_f32       
$suncu -o $NCRES/nvlz4_acs_wht_f32          -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/acs_wht_f32          
$suncu -o $NCRES/nvlz4_hdr_night_f32        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/hdr_night_f32        
$suncu -o $NCRES/nvlz4_hdr_palermo_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/hdr_palermo_f32      
$suncu -o $NCRES/nvlz4_hst_wfc3_uvis_f32    -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/hst_wfc3_uvis_f32    
$suncu -o $NCRES/nvlz4_hst_wfc3_ir_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/hst_wfc3_ir_f32      
$suncu -o $NCRES/nvlz4_spitzer_irac_f32     -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/spitzer_irac_f32     
$suncu -o $NCRES/nvlz4_g24_78_usb2_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/g24_78_usb2_f32      
$suncu -o $NCRES/nvlz4_jw_mirimage_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/jw_mirimage_f32      
$suncu -o $NCRES/nvlz4_tpch_order_f64       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpch_order_f64       
$suncu -o $NCRES/nvlz4_tpcxbb_store_f64     -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcxbb_store_f64     
$suncu -o $NCRES/nvlz4_tpcxbb_web_f64       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcxbb_web_f64       
$suncu -o $NCRES/nvlz4_tpch_lineitem_f32    -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpch_lineitem_f32    
$suncu -o $NCRES/nvlz4_tpcds_catalog_f32    -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcds_catalog_f32    
$suncu -o $NCRES/nvlz4_tpcds_store_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcds_store_f32      
$suncu -o $NCRES/nvlz4_tpcds_web_f32        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_lz4_chunked -f $DATADIR/tpcds_web_f32        
# ================
# test decompress
# ================
$suncu -o $NCRES/nvbitcomp_msg_bt_f64           -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/msg_bt_f64           
$suncu -o $NCRES/nvbitcomp_num_brain_f64        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/num_brain_f64        
$suncu -o $NCRES/nvbitcomp_num_control_f64      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/num_control_f64      
$suncu -o $NCRES/nvbitcomp_rsim_f32             -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/rsim_f32             
$suncu -o $NCRES/nvbitcomp_astro_mhd_f64        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/astro_mhd_f64        
$suncu -o $NCRES/nvbitcomp_astro_pt_f64         -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/astro_pt_f64         
$suncu -o $NCRES/nvbitcomp_miranda3d_f32        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/miranda3d_f32        
$suncu -o $NCRES/nvbitcomp_turbulence_f32       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/turbulence_f32       
$suncu -o $NCRES/nvbitcomp_wave_f32             -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/wave_f32             
$suncu -o $NCRES/nvbitcomp_h3d_temp_f32         -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/h3d_temp_f32         
$suncu -o $NCRES/nvbitcomp_citytemp_f32         -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/citytemp_f32         
$suncu -o $NCRES/nvbitcomp_ts_gas_f32           -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/ts_gas_f32           
$suncu -o $NCRES/nvbitcomp_phone_gyro_f64       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/phone_gyro_f64       
$suncu -o $NCRES/nvbitcomp_wesad_chest_f64      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/wesad_chest_f64      
$suncu -o $NCRES/nvbitcomp_jane_street_f64      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/jane_street_f64      
$suncu -o $NCRES/nvbitcomp_nyc_taxi2015_f64     -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/nyc_taxi2015_f64     
$suncu -o $NCRES/nvbitcomp_spain_gas_price_f64  -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/spain_gas_price_f64  
$suncu -o $NCRES/nvbitcomp_solar_wind_f32       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/solar_wind_f32       
$suncu -o $NCRES/nvbitcomp_acs_wht_f32          -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/acs_wht_f32          
$suncu -o $NCRES/nvbitcomp_hdr_night_f32        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hdr_night_f32        
$suncu -o $NCRES/nvbitcomp_hdr_palermo_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hdr_palermo_f32      
$suncu -o $NCRES/nvbitcomp_hst_wfc3_uvis_f32    -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hst_wfc3_uvis_f32    
$suncu -o $NCRES/nvbitcomp_hst_wfc3_ir_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/hst_wfc3_ir_f32      
$suncu -o $NCRES/nvbitcomp_spitzer_irac_f32     -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/spitzer_irac_f32     
$suncu -o $NCRES/nvbitcomp_g24_78_usb2_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/g24_78_usb2_f32      
$suncu -o $NCRES/nvbitcomp_jw_mirimage_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/jw_mirimage_f32      
$suncu -o $NCRES/nvbitcomp_tpch_order_f64       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpch_order_f64       
$suncu -o $NCRES/nvbitcomp_tpcxbb_store_f64     -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcxbb_store_f64     
$suncu -o $NCRES/nvbitcomp_tpcxbb_web_f64       -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcxbb_web_f64       
$suncu -o $NCRES/nvbitcomp_tpch_lineitem_f32    -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpch_lineitem_f32    
$suncu -o $NCRES/nvbitcomp_tpcds_catalog_f32    -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcds_catalog_f32    
$suncu -o $NCRES/nvbitcomp_tpcds_store_f32      -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcds_store_f32      
$suncu -o $NCRES/nvbitcomp_tpcds_web_f32        -f --set detailed --section SpeedOfLight_RooflineChart $EXECDIR/benchmark_bitcomp_chunked -f $DATADIR/tpcds_web_f32        
