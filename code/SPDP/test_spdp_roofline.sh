#!/bin/bash
# ============== #
# test pFPC #
# ============== #
DATADIR=/home/cc/data
OUTDIR=/home/cc/output
RESDIR=/home/cc/experiments
export ADVIXE_EXPERIMENTAL=intops_strict
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_msg_bt_f64          -- ./spdp 10 < $DATADIR/msg_bt_f64          > $OUTDIR/msg_bt_f64.spdp          
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_num_brain_f64       -- ./spdp 10 < $DATADIR/num_brain_f64       > $OUTDIR/num_brain_f64.spdp       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_num_control_f64     -- ./spdp 10 < $DATADIR/num_control_f64     > $OUTDIR/num_control_f64.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_rsim_f32            -- ./spdp 10 < $DATADIR/rsim_f32            > $OUTDIR/rsim_f32.spdp            
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_astro_mhd_f64       -- ./spdp 10 < $DATADIR/astro_mhd_f64       > $OUTDIR/astro_mhd_f64.spdp       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_astro_pt_f64        -- ./spdp 10 < $DATADIR/astro_pt_f64        > $OUTDIR/astro_pt_f64.spdp        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_miranda3d_f32       -- ./spdp 10 < $DATADIR/miranda3d_f32       > $OUTDIR/miranda3d_f32.spdp       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_turbulence_f32      -- ./spdp 10 < $DATADIR/turbulence_f32      > $OUTDIR/turbulence_f32.spdp      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_wave_f32            -- ./spdp 10 < $DATADIR/wave_f32            > $OUTDIR/wave_f32.spdp            
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_h3d_temp_f32        -- ./spdp 10 < $DATADIR/h3d_temp_f32        > $OUTDIR/h3d_temp_f32.spdp        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_citytemp_f32        -- ./spdp 10 < $DATADIR/citytemp_f32        > $OUTDIR/citytemp_f32.spdp        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_ts_gas_f32          -- ./spdp 10 < $DATADIR/ts_gas_f32          > $OUTDIR/ts_gas_f32.spdp          
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_phone_gyro_f64      -- ./spdp 10 < $DATADIR/phone_gyro_f64      > $OUTDIR/phone_gyro_f64.spdp      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_wesad_chest_f64     -- ./spdp 10 < $DATADIR/wesad_chest_f64     > $OUTDIR/wesad_chest_f64.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_jane_street_f64     -- ./spdp 10 < $DATADIR/jane_street_f64     > $OUTDIR/jane_street_f64.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_nyc_taxi2015_f64    -- ./spdp 10 < $DATADIR/nyc_taxi2015_f64    > $OUTDIR/nyc_taxi2015_f64.spdp    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_spain_gas_price_f64 -- ./spdp 10 < $DATADIR/spain_gas_price_f64 > $OUTDIR/spain_gas_price_f64.spdp 
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_solar_wind_f32      -- ./spdp 10 < $DATADIR/solar_wind_f32      > $OUTDIR/solar_wind_f32.spdp      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_acs_wht_f32         -- ./spdp 10 < $DATADIR/acs_wht_f32         > $OUTDIR/acs_wht_f32.spdp         
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_hdr_night_f32       -- ./spdp 10 < $DATADIR/hdr_night_f32       > $OUTDIR/hdr_night_f32.spdp       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_hdr_palermo_f32     -- ./spdp 10 < $DATADIR/hdr_palermo_f32     > $OUTDIR/hdr_palermo_f32.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_hst_wfc3_uvis_f32   -- ./spdp 10 < $DATADIR/hst_wfc3_uvis_f32   > $OUTDIR/hst_wfc3_uvis_f32.spdp   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_hst_wfc3_ir_f32     -- ./spdp 10 < $DATADIR/hst_wfc3_ir_f32     > $OUTDIR/hst_wfc3_ir_f32.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_spitzer_irac_f32    -- ./spdp 10 < $DATADIR/spitzer_irac_f32    > $OUTDIR/spitzer_irac_f32.spdp    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_g24_78_usb2_f32     -- ./spdp 10 < $DATADIR/g24_78_usb2_f32     > $OUTDIR/g24_78_usb2_f32.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_jw_mirimage_f32     -- ./spdp 10 < $DATADIR/jw_mirimage_f32     > $OUTDIR/jw_mirimage_f32.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpch_order_f64      -- ./spdp 10 < $DATADIR/tpch_order_f64      > $OUTDIR/tpch_order_f64.spdp      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpcxbb_store_f64    -- ./spdp 10 < $DATADIR/tpcxbb_store_f64    > $OUTDIR/tpcxbb_store_f64.spdp    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpcxbb_web_f64      -- ./spdp 10 < $DATADIR/tpcxbb_web_f64      > $OUTDIR/tpcxbb_web_f64.spdp      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpch_lineitem_f32   -- ./spdp 10 < $DATADIR/tpch_lineitem_f32   > $OUTDIR/tpch_lineitem_f32.spdp   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpcds_catalog_f32   -- ./spdp 10 < $DATADIR/tpcds_catalog_f32   > $OUTDIR/tpcds_catalog_f32.spdp   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpcds_store_f32     -- ./spdp 10 < $DATADIR/tpcds_store_f32     > $OUTDIR/tpcds_store_f32.spdp     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_comp_tpcds_web_f32       -- ./spdp 10 < $DATADIR/tpcds_web_f32       > $OUTDIR/tpcds_web_f32.spdp       
# ================
# test decompress
# ================
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_msg_bt_f64          -- ./spdp < $OUTDIR/msg_bt_f64.spdp          > $DATADIR/msg_bt_f64.spdpout          
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_num_brain_f64       -- ./spdp < $OUTDIR/num_brain_f64.spdp       > $DATADIR/num_brain_f64.spdpout       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_num_control_f64     -- ./spdp < $OUTDIR/num_control_f64.spdp     > $DATADIR/num_control_f64.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_rsim_f32            -- ./spdp < $OUTDIR/rsim_f32.spdp            > $DATADIR/rsim_f32.spdpout            
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_astro_mhd_f64       -- ./spdp < $OUTDIR/astro_mhd_f64.spdp       > $DATADIR/astro_mhd_f64.spdpout       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_astro_pt_f64        -- ./spdp < $OUTDIR/astro_pt_f64.spdp        > $DATADIR/astro_pt_f64.spdpout        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_miranda3d_f32       -- ./spdp < $OUTDIR/miranda3d_f32.spdp       > $DATADIR/miranda3d_f32.spdpout       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_turbulence_f32      -- ./spdp < $OUTDIR/turbulence_f32.spdp      > $DATADIR/turbulence_f32.spdpout      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_wave_f32            -- ./spdp < $OUTDIR/wave_f32.spdp            > $DATADIR/wave_f32.spdpout            
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_h3d_temp_f32        -- ./spdp < $OUTDIR/h3d_temp_f32.spdp        > $DATADIR/h3d_temp_f32.spdpout        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_citytemp_f32        -- ./spdp < $OUTDIR/citytemp_f32.spdp        > $DATADIR/citytemp_f32.spdpout        
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_ts_gas_f32          -- ./spdp < $OUTDIR/ts_gas_f32.spdp          > $DATADIR/ts_gas_f32.spdpout          
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_phone_gyro_f64      -- ./spdp < $OUTDIR/phone_gyro_f64.spdp      > $DATADIR/phone_gyro_f64.spdpout      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_wesad_chest_f64     -- ./spdp < $OUTDIR/wesad_chest_f64.spdp     > $DATADIR/wesad_chest_f64.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_jane_street_f64     -- ./spdp < $OUTDIR/jane_street_f64.spdp     > $DATADIR/jane_street_f64.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_nyc_taxi2015_f64    -- ./spdp < $OUTDIR/nyc_taxi2015_f64.spdp    > $DATADIR/nyc_taxi2015_f64.spdpout    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_spain_gas_price_f64 -- ./spdp < $OUTDIR/spain_gas_price_f64.spdp > $DATADIR/spain_gas_price_f64.spdpout 
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_solar_wind_f32      -- ./spdp < $OUTDIR/solar_wind_f32.spdp      > $DATADIR/solar_wind_f32.spdpout      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_acs_wht_f32         -- ./spdp < $OUTDIR/acs_wht_f32.spdp         > $DATADIR/acs_wht_f32.spdpout         
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_hdr_night_f32       -- ./spdp < $OUTDIR/hdr_night_f32.spdp       > $DATADIR/hdr_night_f32.spdpout       
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_hdr_palermo_f32     -- ./spdp < $OUTDIR/hdr_palermo_f32.spdp     > $DATADIR/hdr_palermo_f32.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_hst_wfc3_uvis_f32   -- ./spdp < $OUTDIR/hst_wfc3_uvis_f32.spdp   > $DATADIR/hst_wfc3_uvis_f32.spdpout   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_hst_wfc3_ir_f32     -- ./spdp < $OUTDIR/hst_wfc3_ir_f32.spdp     > $DATADIR/hst_wfc3_ir_f32.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_spitzer_irac_f32    -- ./spdp < $OUTDIR/spitzer_irac_f32.spdp    > $DATADIR/spitzer_irac_f32.spdpout    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_g24_78_usb2_f32     -- ./spdp < $OUTDIR/g24_78_usb2_f32.spdp     > $DATADIR/g24_78_usb2_f32.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_jw_mirimage_f32     -- ./spdp < $OUTDIR/jw_mirimage_f32.spdp     > $DATADIR/jw_mirimage_f32.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpch_order_f64      -- ./spdp < $OUTDIR/tpch_order_f64.spdp      > $DATADIR/tpch_order_f64.spdpout      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpcxbb_store_f64    -- ./spdp < $OUTDIR/tpcxbb_store_f64.spdp    > $DATADIR/tpcxbb_store_f64.spdpout    
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpcxbb_web_f64      -- ./spdp < $OUTDIR/tpcxbb_web_f64.spdp      > $DATADIR/tpcxbb_web_f64.spdpout      
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpch_lineitem_f32   -- ./spdp < $OUTDIR/tpch_lineitem_f32.spdp   > $DATADIR/tpch_lineitem_f32.spdpout   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpcds_catalog_f32   -- ./spdp < $OUTDIR/tpcds_catalog_f32.spdp   > $DATADIR/tpcds_catalog_f32.spdpout   
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpcds_store_f32     -- ./spdp < $OUTDIR/tpcds_store_f32.spdp     > $DATADIR/tpcds_store_f32.spdpout     
advixe-cl --collect=roofline --interval=10 --project-dir=/home/cc/advi/spdp_decomp_tpcds_web_f32       -- ./spdp < $OUTDIR/tpcds_web_f32.spdp       > $DATADIR/tpcds_web_f32.spdpout       
rm $DATADIR/*.spdpout $OUTDIR/*.spdp