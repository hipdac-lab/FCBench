# ============== #
# test ndzip #
# ============== #
EXECDIR=/home/cc/code
DATADIR=/home/cc/data
OUTDIR=/home/cc/output
RESDIR=/home/cc/experiments
export ADVIXE_EXPERIMENTAL=roofline
# ================ #
# test ndzip-cpu #
# ================ #
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_msg_bt_f64          --  $EXECDIR/ndzip/build/compress -i $DATADIR/msg_bt_f64          -o $OUTDIR/msg_bt_f64.ndzip    		 -t double -n 33298679       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_num_brain_f64       --  $EXECDIR/ndzip/build/compress -i $DATADIR/num_brain_f64       -o $OUTDIR/num_brain_f64.ndzip       -t double -n 17730000       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_num_control_f64     --  $EXECDIR/ndzip/build/compress -i $DATADIR/num_control_f64     -o $OUTDIR/num_control_f64.ndzip     -t double -n 19938093       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_rsim_f32            --  $EXECDIR/ndzip/build/compress -i $DATADIR/rsim_f32            -o $OUTDIR/rsim_f32.ndzip            -t float  -n 2048 11509     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_astro_mhd_f64       --  $EXECDIR/ndzip/build/compress -i $DATADIR/astro_mhd_f64       -o $OUTDIR/astro_mhd_f64.ndzip       -t double -n 130 514 1026   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_astro_pt_f64        --  $EXECDIR/ndzip/build/compress -i $DATADIR/astro_pt_f64        -o $OUTDIR/astro_pt_f64.ndzip        -t double -n 512 256 640    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_miranda3d_f32       --  $EXECDIR/ndzip/build/compress -i $DATADIR/miranda3d_f32       -o $OUTDIR/miranda3d_f32.ndzip       -t float  -n 1024 1024 1024 
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_turbulence_f32      --  $EXECDIR/ndzip/build/compress -i $DATADIR/turbulence_f32      -o $OUTDIR/turbulence_f32.ndzip      -t float  -n 256 256 256    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_wave_f32            --  $EXECDIR/ndzip/build/compress -i $DATADIR/wave_f32            -o $OUTDIR/wave_f32.ndzip            -t float  -n 512 512 512    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_h3d_temp_f32        --  $EXECDIR/ndzip/build/compress -i $DATADIR/h3d_temp_f32        -o $OUTDIR/h3d_temp_f32.ndzip        -t float  -n 100 500 500    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_citytemp_f32        --  $EXECDIR/ndzip/build/compress -i $DATADIR/citytemp_f32        -o $OUTDIR/citytemp_f32.ndzip        -t float  -n 2906326        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_ts_gas_f32          --  $EXECDIR/ndzip/build/compress -i $DATADIR/ts_gas_f32          -o $OUTDIR/ts_gas_f32.ndzip          -t float  -n 76863200       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_phone_gyro_f64      --  $EXECDIR/ndzip/build/compress -i $DATADIR/phone_gyro_f64      -o $OUTDIR/phone_gyro_f64.ndzip      -t double -n 13932632 3     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_wesad_chest_f64     --  $EXECDIR/ndzip/build/compress -i $DATADIR/wesad_chest_f64     -o $OUTDIR/wesad_chest_f64.ndzip     -t double -n 4255300 8      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_jane_street_f64     --  $EXECDIR/ndzip/build/compress -i $DATADIR/jane_street_f64     -o $OUTDIR/jane_street_f64.ndzip     -t double -n 1664520 136    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_nyc_taxi2015_f64    --  $EXECDIR/ndzip/build/compress -i $DATADIR/nyc_taxi2015_f64    -o $OUTDIR/nyc_taxi2015_f64.ndzip    -t double -n 12748986 7     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_spain_gas_price_f64 --  $EXECDIR/ndzip/build/compress -i $DATADIR/spain_gas_price_f64 -o $OUTDIR/spain_gas_price_f64.ndzip -t double -n 36942486 3     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_solar_wind_f32      --  $EXECDIR/ndzip/build/compress -i $DATADIR/solar_wind_f32      -o $OUTDIR/solar_wind_f32.ndzip      -t float  -n 7571081 14     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_acs_wht_f32         --  $EXECDIR/ndzip/build/compress -i $DATADIR/acs_wht_f32         -o $OUTDIR/acs_wht_f32.ndzip         -t float  -n 7500 7500      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_hdr_night_f32       --  $EXECDIR/ndzip/build/compress -i $DATADIR/hdr_night_f32       -o $OUTDIR/hdr_night_f32.ndzip       -t float  -n 8192 16384     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_hdr_palermo_f32     --  $EXECDIR/ndzip/build/compress -i $DATADIR/hdr_palermo_f32     -o $OUTDIR/hdr_palermo_f32.ndzip     -t float  -n 10268 20536    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_hst_wfc3_uvis_f32   --  $EXECDIR/ndzip/build/compress -i $DATADIR/hst_wfc3_uvis_f32   -o $OUTDIR/hst_wfc3_uvis_f32.ndzip   -t float  -n 5329 5110      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_hst_wfc3_ir_f32     --  $EXECDIR/ndzip/build/compress -i $DATADIR/hst_wfc3_ir_f32     -o $OUTDIR/hst_wfc3_ir_f32.ndzip     -t float  -n 2484 2417      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_spitzer_irac_f32    --  $EXECDIR/ndzip/build/compress -i $DATADIR/spitzer_irac_f32    -o $OUTDIR/spitzer_irac_f32.ndzip    -t float  -n 6456 6389      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_g24_78_usb2_f32     --  $EXECDIR/ndzip/build/compress -i $DATADIR/g24_78_usb2_f32     -o $OUTDIR/g24_78_usb2_f32.ndzip     -t float  -n 2426 371 371   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_jw_mirimage_f32     --  $EXECDIR/ndzip/build/compress -i $DATADIR/jw_mirimage_f32     -o $OUTDIR/jw_mirimage_f32.ndzip     -t float  -n 40 1024 1032   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpch_order_f64      --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpch_order_f64      -o $OUTDIR/tpch_order_f64.ndzip      -t double -n 15000000       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpcxbb_store_f64    --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpcxbb_store_f64    -o $OUTDIR/tpcxbb_store_f64.ndzip    -t double -n 8228343 12     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpcxbb_web_f64      --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpcxbb_web_f64      -o $OUTDIR/tpcxbb_web_f64.ndzip      -t double -n 8223189 15     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpch_lineitem_f32   --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpch_lineitem_f32   -o $OUTDIR/tpch_lineitem_f32.ndzip   -t float  -n 59986051 4     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpcds_catalog_f32   --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpcds_catalog_f32   -o $OUTDIR/tpcds_catalog_f32.ndzip   -t float  -n 2880058 15     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpcds_store_f32     --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpcds_store_f32     -o $OUTDIR/tpcds_store_f32.ndzip     -t float  -n 5760749 12     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_comp_tpcds_web_f32       --  $EXECDIR/ndzip/build/compress -i $DATADIR/tpcds_web_f32       -o $OUTDIR/tpcds_web_f32.ndzip       -t float  -n 1439247 15     

echo "test ndzip-cpu decompress"
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_msg_bt_f64          -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/msg_bt_f64.ndzip    		-o $DATADIR/msg_bt_f64.ndzout          -t double -n 33298679     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_num_brain_f64       -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/num_brain_f64.ndzip       -o $DATADIR/num_brain_f64.ndzout       -t double -n 17730000       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_num_control_f64     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/num_control_f64.ndzip     -o $DATADIR/num_control_f64.ndzout     -t double -n 19938093       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_rsim_f32            -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/rsim_f32.ndzip            -o $DATADIR/rsim_f32.ndzout            -t float  -n 2048 11509     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_astro_mhd_f64       -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/astro_mhd_f64.ndzip       -o $DATADIR/astro_mhd_f64.ndzout       -t double -n 130 514 1026   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_astro_pt_f64        -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/astro_pt_f64.ndzip        -o $DATADIR/astro_pt_f64.ndzout        -t double -n 512 256 640    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_miranda3d_f32       -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/miranda3d_f32.ndzip       -o $DATADIR/miranda3d_f32.ndzout       -t float  -n 1024 1024 1024 
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_turbulence_f32      -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/turbulence_f32.ndzip      -o $DATADIR/turbulence_f32.ndzout      -t float  -n 256 256 256    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_wave_f32            -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/wave_f32.ndzip            -o $DATADIR/wave_f32.ndzout            -t float  -n 512 512 512    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_h3d_temp_f32        -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/h3d_temp_f32.ndzip        -o $DATADIR/h3d_temp_f32.ndzout        -t float  -n 100 500 500    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_citytemp_f32        -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/citytemp_f32.ndzip        -o $DATADIR/citytemp_f32.ndzout        -t float  -n 2906326        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_ts_gas_f32          -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/ts_gas_f32.ndzip          -o $DATADIR/ts_gas_f32.ndzout          -t float  -n 76863200       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_phone_gyro_f64      -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/phone_gyro_f64.ndzip      -o $DATADIR/phone_gyro_f64.ndzout      -t double -n 13932632 3     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_wesad_chest_f64     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/wesad_chest_f64.ndzip     -o $DATADIR/wesad_chest_f64.ndzout     -t double -n 4255300 8      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_jane_street_f64     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/jane_street_f64.ndzip     -o $DATADIR/jane_street_f64.ndzout     -t double -n 1664520 136    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_nyc_taxi2015_f64    -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/nyc_taxi2015_f64.ndzip    -o $DATADIR/nyc_taxi2015_f64.ndzout    -t double -n 12748986 7     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_spain_gas_price_f64 -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/spain_gas_price_f64.ndzip -o $DATADIR/spain_gas_price_f64.ndzout -t double -n 36942486 3     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_solar_wind_f32      -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/solar_wind_f32.ndzip      -o $DATADIR/solar_wind_f32.ndzout      -t float  -n 7571081 14     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_acs_wht_f32         -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/acs_wht_f32.ndzip         -o $DATADIR/acs_wht_f32.ndzout         -t float  -n 7500 7500      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_hdr_night_f32       -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/hdr_night_f32.ndzip       -o $DATADIR/hdr_night_f32.ndzout       -t float  -n 8192 16384     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_hdr_palermo_f32     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/hdr_palermo_f32.ndzip     -o $DATADIR/hdr_palermo_f32.ndzout     -t float  -n 10268 20536    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_hst_wfc3_uvis_f32   -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/hst_wfc3_uvis_f32.ndzip   -o $DATADIR/hst_wfc3_uvis_f32.ndzout   -t float  -n 5329 5110      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_hst_wfc3_ir_f32     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/hst_wfc3_ir_f32.ndzip     -o $DATADIR/hst_wfc3_ir_f32.ndzout     -t float  -n 2484 2417      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_spitzer_irac_f32    -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/spitzer_irac_f32.ndzip    -o $DATADIR/spitzer_irac_f32.ndzout    -t float  -n 6456 6389      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_g24_78_usb2_f32     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/g24_78_usb2_f32.ndzip     -o $DATADIR/g24_78_usb2_f32.ndzout     -t float  -n 2426 371 371   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_jw_mirimage_f32     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/jw_mirimage_f32.ndzip     -o $DATADIR/jw_mirimage_f32.ndzout     -t float  -n 40 1024 1032   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpch_order_f64      -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpch_order_f64.ndzip      -o $DATADIR/tpch_order_f64.ndzout      -t double -n 15000000       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpcxbb_store_f64    -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpcxbb_store_f64.ndzip    -o $DATADIR/tpcxbb_store_f64.ndzout    -t double -n 8228343 12     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpcxbb_web_f64      -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpcxbb_web_f64.ndzip      -o $DATADIR/tpcxbb_web_f64.ndzout      -t double -n 8223189 15     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpch_lineitem_f32   -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpch_lineitem_f32.ndzip   -o $DATADIR/tpch_lineitem_f32.ndzout   -t float  -n 59986051 4     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpcds_catalog_f32   -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpcds_catalog_f32.ndzip   -o $DATADIR/tpcds_catalog_f32.ndzout   -t float  -n 2880058 15     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpcds_store_f32     -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpcds_store_f32.ndzip     -o $DATADIR/tpcds_store_f32.ndzout     -t float  -n 5760749 12     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/ndzipc_decomp_tpcds_web_f32       -- $EXECDIR/ndzip/build/compress -d -i $OUTDIR/tpcds_web_f32.ndzip       -o $DATADIR/tpcds_web_f32.ndzout       -t float  -n 1439247 15     
# ls -l $OUTDIR/*.ndzip
# ls -l $DATADIR/*.ndzout
rm $OUTDIR/*.ndzip $DATADIR/*.ndzout

