#!/bin/bash
DATADIR=/home/cc/data
OUTDIR=/home/cc/output
RESDIR=/home/cc/experiments
NCRES=/home/cc/ncres
suncu='sudo /usr/local/cuda-11.2/bin/ncu'
# ================ #
# test GFC #
# ================ #
./GFC 32 32 1 < $DATADIR/msg_bt_f64           > $OUTDIR/msg_bt_f64.gfc           
./GFC 32 32 1 < $DATADIR/num_brain_f64        > $OUTDIR/num_brain_f64.gfc        
./GFC 32 32 1 < $DATADIR/num_control_f64      > $OUTDIR/num_control_f64.gfc      
./GFC 32 32 2 < $DATADIR/rsim_f32             > $OUTDIR/rsim_f32.gfc             
./GFC 32 32 3 < $DATADIR/turbulence_f32       > $OUTDIR/turbulence_f32.gfc       
./GFC 32 32 3 < $DATADIR/wave_f32             > $OUTDIR/wave_f32.gfc             
./GFC 32 32 3 < $DATADIR/h3d_temp_f32         > $OUTDIR/h3d_temp_f32.gfc         
./GFC 32 32 1 < $DATADIR/citytemp_f32         > $OUTDIR/citytemp_f32.gfc         
./GFC 32 32 1 < $DATADIR/ts_gas_f32           > $OUTDIR/ts_gas_f32.gfc           
./GFC 32 32 2 < $DATADIR/phone_gyro_f64       > $OUTDIR/phone_gyro_f64.gfc       
./GFC 32 32 2 < $DATADIR/wesad_chest_f64      > $OUTDIR/wesad_chest_f64.gfc      
./GFC 32 32 2 < $DATADIR/solar_wind_f32       > $OUTDIR/solar_wind_f32.gfc       
./GFC 32 32 2 < $DATADIR/acs_wht_f32          > $OUTDIR/acs_wht_f32.gfc          
./GFC 32 32 2 < $DATADIR/hdr_night_f32        > $OUTDIR/hdr_night_f32.gfc        
./GFC 32 32 2 < $DATADIR/hst_wfc3_uvis_f32    > $OUTDIR/hst_wfc3_uvis_f32.gfc    
./GFC 32 32 2 < $DATADIR/hst_wfc3_ir_f32      > $OUTDIR/hst_wfc3_ir_f32.gfc      
./GFC 32 32 2 < $DATADIR/spitzer_irac_f32     > $OUTDIR/spitzer_irac_f32.gfc     
./GFC 32 32 3 < $DATADIR/jw_mirimage_f32      > $OUTDIR/jw_mirimage_f32.gfc      
./GFC 32 32 1 < $DATADIR/tpch_order_f64       > $OUTDIR/tpch_order_f64.gfc       
./GFC 32 32 2 < $DATADIR/tpcds_catalog_f32    > $OUTDIR/tpcds_catalog_f32.gfc    
./GFC 32 32 2 < $DATADIR/tpcds_store_f32      > $OUTDIR/tpcds_store_f32.gfc      
./GFC 32 32 2 < $DATADIR/tpcds_web_f32        > $OUTDIR/tpcds_web_f32.gfc        
# ================
# test decompress
# ================
# $suncu -o $NCRES/gfc_decomp_msg_bt_f64        -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/msg_bt_f64.gfc          > $DATADIR/msg_bt_f64.gfcout             
# $suncu -o $NCRES/gfc_decomp_num_brain_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/num_brain_f64.gfc       > $DATADIR/num_brain_f64.gfcout          
# $suncu -o $NCRES/gfc_decomp_num_control_f64   -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/num_control_f64.gfc     > $DATADIR/num_control_f64.gfcout        
# $suncu -o $NCRES/gfc_decomp_rsim_f32          -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/rsim_f32.gfc            > $DATADIR/rsim_f32.gfcout               
# $suncu -o $NCRES/gfc_decomp_turbulence_f32    -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/turbulence_f32.gfc      > $DATADIR/turbulence_f32.gfcout         
# $suncu -o $NCRES/gfc_decomp_wave_f32          -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/wave_f32.gfc            > $DATADIR/wave_f32.gfcout               
# $suncu -o $NCRES/gfc_decomp_h3d_temp_f32      -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/h3d_temp_f32.gfc        > $DATADIR/h3d_temp_f32.gfcout           
# $suncu -o $NCRES/gfc_decomp_citytemp_f32      -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/citytemp_f32.gfc        > $DATADIR/citytemp_f32.gfcout           
# $suncu -o $NCRES/gfc_decomp_ts_gas_f32        -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/ts_gas_f32.gfc          > $DATADIR/ts_gas_f32.gfcout             
# $suncu -o $NCRES/gfc_decomp_phone_gyro_f64    -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/phone_gyro_f64.gfc      > $DATADIR/phone_gyro_f64.gfcout         
# $suncu -o $NCRES/gfc_decomp_wesad_chest_f64   -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/wesad_chest_f64.gfc     > $DATADIR/wesad_chest_f64.gfcout        
# $suncu -o $NCRES/gfc_decomp_solar_wind_f32    -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/solar_wind_f32.gfc      > $DATADIR/solar_wind_f32.gfcout         
# $suncu -o $NCRES/gfc_decomp_acs_wht_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/acs_wht_f32.gfc         > $DATADIR/acs_wht_f32.gfcout            
# $suncu -o $NCRES/gfc_decomp_hdr_night_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/hdr_night_f32.gfc       > $DATADIR/hdr_night_f32.gfcout          
# $suncu -o $NCRES/gfc_decomp_hst_wfc3_uvis_f32 -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/hst_wfc3_uvis_f32.gfc   > $DATADIR/hst_wfc3_uvis_f32.gfcout      
# $suncu -o $NCRES/gfc_decomp_hst_wfc3_ir_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/hst_wfc3_ir_f32.gfc     > $DATADIR/hst_wfc3_ir_f32.gfcout        
# $suncu -o $NCRES/gfc_decomp_spitzer_irac_f32  -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/spitzer_irac_f32.gfc    > $DATADIR/spitzer_irac_f32.gfcout       
# $suncu -o $NCRES/gfc_decomp_jw_mirimage_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/jw_mirimage_f32.gfc     > $DATADIR/jw_mirimage_f32.gfcout        
# $suncu -o $NCRES/gfc_decomp_tpch_order_f64    -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/tpch_order_f64.gfc      > $DATADIR/tpch_order_f64.gfcout         
# $suncu -o $NCRES/gfc_decomp_tpcds_catalog_f32 -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/tpcds_catalog_f32.gfc   > $DATADIR/tpcds_catalog_f32.gfcout      
# $suncu -o $NCRES/gfc_decomp_tpcds_store_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/tpcds_store_f32.gfc     > $DATADIR/tpcds_store_f32.gfcout        
# $suncu -o $NCRES/gfc_decomp_tpcds_web_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./GFC < $OUTDIR/tpcds_web_f32.gfc       > $DATADIR/tpcds_web_f32.gfcout          
# sudo rm $OUTDIR/*.gfc $DATADIR/*.gfcout
