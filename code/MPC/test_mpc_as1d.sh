# ============== #
# test MPC #
# ============== #
DATADIR=/home/cc/data
RESDIR=/home/cc/experiments
# ================ #
# test MPC #
# ================ #
./MPC_double $DATADIR/msg_bt_f64          1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/num_brain_f64       1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/num_control_f64     1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/rsim_f32            1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/astro_mhd_f64       1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/astro_pt_f64        1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/miranda3d_f32       1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/turbulence_f32      1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/wave_f32            1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/h3d_temp_f32        1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/citytemp_f32        1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/ts_gas_f32          1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/phone_gyro_f64      1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/wesad_chest_f64     1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/jane_street_f64     1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/nyc_taxi2015_f64    1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/spain_gas_price_f64 1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/solar_wind_f32      1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/acs_wht_f32         1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/hdr_night_f32       1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/hdr_palermo_f32     1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/hst_wfc3_uvis_f32   1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/hst_wfc3_ir_f32     1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/spitzer_irac_f32    1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/g24_78_usb2_f32     1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/jw_mirimage_f32     1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/tpch_order_f64      1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/tpcxbb_store_f64    1 >> $RESDIR/mpc_comp.txt
./MPC_double $DATADIR/tpcxbb_web_f64      1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/tpch_lineitem_f32   1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/tpcds_catalog_f32   1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/tpcds_store_f32     1 >> $RESDIR/mpc_comp.txt
./MPC_float  $DATADIR/tpcds_web_f32       1 >> $RESDIR/mpc_comp.txt
# ================ #
# test decompress  #
# ================ #
./MPC_double $DATADIR/msg_bt_f64.mpc          >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/num_brain_f64.mpc       >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/num_control_f64.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/rsim_f32.mpc            >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/astro_mhd_f64.mpc       >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/astro_pt_f64.mpc        >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/miranda3d_f32.mpc       >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/turbulence_f32.mpc      >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/wave_f32.mpc            >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/h3d_temp_f32.mpc        >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/citytemp_f32.mpc        >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/ts_gas_f32.mpc          >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/phone_gyro_f64.mpc      >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/wesad_chest_f64.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/jane_street_f64.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/nyc_taxi2015_f64.mpc    >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/spain_gas_price_f64.mpc >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/solar_wind_f32.mpc      >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/acs_wht_f32.mpc         >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/hdr_night_f32.mpc       >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/hdr_palermo_f32.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/hst_wfc3_uvis_f32.mpc   >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/hst_wfc3_ir_f32.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/spitzer_irac_f32.mpc    >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/g24_78_usb2_f32.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/jw_mirimage_f32.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/tpch_order_f64.mpc      >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/tpcxbb_store_f64.mpc    >> $RESDIR/mpc_decomp.txt
./MPC_double $DATADIR/tpcxbb_web_f64.mpc      >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/tpch_lineitem_f32.mpc   >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/tpcds_catalog_f32.mpc   >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/tpcds_store_f32.mpc     >> $RESDIR/mpc_decomp.txt
./MPC_float  $DATADIR/tpcds_web_f32.mpc       >> $RESDIR/mpc_decomp.txt
rm $DATADIR/*.mpc
diff $DATADIR/msg_bt_f64          $DATADIR/msg_bt_f64.mpc.org        
diff $DATADIR/num_brain_f64       $DATADIR/num_brain_f64.mpc.org     
diff $DATADIR/num_control_f64     $DATADIR/num_control_f64.mpc.org   
diff $DATADIR/rsim_f32            $DATADIR/rsim_f32.mpc.org          
diff $DATADIR/astro_mhd_f64       $DATADIR/astro_mhd_f64.mpc.org     
diff $DATADIR/astro_pt_f64        $DATADIR/astro_pt_f64.mpc.org      
diff $DATADIR/miranda3d_f32       $DATADIR/miranda3d_f32.mpc.org     
diff $DATADIR/turbulence_f32      $DATADIR/turbulence_f32.mpc.org    
diff $DATADIR/wave_f32            $DATADIR/wave_f32.mpc.org          
diff $DATADIR/h3d_temp_f32        $DATADIR/h3d_temp_f32.mpc.org      
diff $DATADIR/citytemp_f32        $DATADIR/citytemp_f32.mpc.org      
diff $DATADIR/ts_gas_f32          $DATADIR/ts_gas_f32.mpc.org        
diff $DATADIR/phone_gyro_f64      $DATADIR/phone_gyro_f64.mpc.org    
diff $DATADIR/wesad_chest_f64     $DATADIR/wesad_chest_f64.mpc.org   
diff $DATADIR/jane_street_f64     $DATADIR/jane_street_f64.mpc.org   
diff $DATADIR/nyc_taxi2015_f64    $DATADIR/nyc_taxi2015_f64.mpc.org  
diff $DATADIR/spain_gas_price_f64 $DATADIR/spain_gas_price_f64.mpc.org
diff $DATADIR/solar_wind_f32      $DATADIR/solar_wind_f32.mpc.org    
diff $DATADIR/acs_wht_f32         $DATADIR/acs_wht_f32.mpc.org       
diff $DATADIR/hdr_night_f32       $DATADIR/hdr_night_f32.mpc.org     
diff $DATADIR/hdr_palermo_f32     $DATADIR/hdr_palermo_f32.mpc.org   
diff $DATADIR/hst_wfc3_uvis_f32   $DATADIR/hst_wfc3_uvis_f32.mpc.org 
diff $DATADIR/hst_wfc3_ir_f32     $DATADIR/hst_wfc3_ir_f32.mpc.org   
diff $DATADIR/spitzer_irac_f32    $DATADIR/spitzer_irac_f32.mpc.org  
diff $DATADIR/g24_78_usb2_f32     $DATADIR/g24_78_usb2_f32.mpc.org   
diff $DATADIR/jw_mirimage_f32     $DATADIR/jw_mirimage_f32.mpc.org   
diff $DATADIR/tpch_order_f64      $DATADIR/tpch_order_f64.mpc.org    
diff $DATADIR/tpcxbb_store_f64    $DATADIR/tpcxbb_store_f64.mpc.org  
diff $DATADIR/tpcxbb_web_f64      $DATADIR/tpcxbb_web_f64.mpc.org    
diff $DATADIR/tpch_lineitem_f32   $DATADIR/tpch_lineitem_f32.mpc.org 
diff $DATADIR/tpcds_catalog_f32   $DATADIR/tpcds_catalog_f32.mpc.org 
diff $DATADIR/tpcds_store_f32     $DATADIR/tpcds_store_f32.mpc.org   
diff $DATADIR/tpcds_web_f32       $DATADIR/tpcds_web_f32.mpc.org     
rm $DATADIR/*.mpc.org