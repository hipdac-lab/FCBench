# ============== #
# test MPC #
# ============== #
DATADIR=/home/cc/data
RESDIR=/home/cc/experiments
NCRES=/home/cc/ncres
suncu='sudo /usr/local/cuda-11.2/bin/ncu'
# ================ #
# test MPC #
# ================ #
$suncu -o $NCRES/mpc_comp_fmsg_bt_f64          -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/msg_bt_f64          1 
$suncu -o $NCRES/mpc_comp_fnum_brain_f64       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/num_brain_f64       1 
$suncu -o $NCRES/mpc_comp_fnum_control_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/num_control_f64     1 
$suncu -o $NCRES/mpc_comp_frsim_f32            -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/rsim_f32            2 
$suncu -o $NCRES/mpc_comp_fastro_mhd_f64       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/astro_mhd_f64       3 
$suncu -o $NCRES/mpc_comp_fastro_pt_f64        -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/astro_pt_f64        3 
$suncu -o $NCRES/mpc_comp_fmiranda3d_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/miranda3d_f32       3 
$suncu -o $NCRES/mpc_comp_fturbulence_f32      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/turbulence_f32      3 
$suncu -o $NCRES/mpc_comp_fwave_f32            -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/wave_f32            3 
$suncu -o $NCRES/mpc_comp_fh3d_temp_f32        -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/h3d_temp_f32        3 
$suncu -o $NCRES/mpc_comp_fcitytemp_f32        -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/citytemp_f32        1 
$suncu -o $NCRES/mpc_comp_fts_gas_f32          -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/ts_gas_f32          1 
$suncu -o $NCRES/mpc_comp_fphone_gyro_f64      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/phone_gyro_f64      2 
$suncu -o $NCRES/mpc_comp_fwesad_chest_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/wesad_chest_f64     2 
$suncu -o $NCRES/mpc_comp_fjane_street_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/jane_street_f64     2 
$suncu -o $NCRES/mpc_comp_fnyc_taxi2015_f64    -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/nyc_taxi2015_f64    2 
$suncu -o $NCRES/mpc_comp_fspain_gas_price_f64 -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/spain_gas_price_f64 2 
$suncu -o $NCRES/mpc_comp_fsolar_wind_f32      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/solar_wind_f32      2 
$suncu -o $NCRES/mpc_comp_facs_wht_f32         -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/acs_wht_f32         2 
$suncu -o $NCRES/mpc_comp_fhdr_night_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hdr_night_f32       2 
$suncu -o $NCRES/mpc_comp_fhdr_palermo_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hdr_palermo_f32     2 
$suncu -o $NCRES/mpc_comp_fhst_wfc3_uvis_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hst_wfc3_uvis_f32   2 
$suncu -o $NCRES/mpc_comp_fhst_wfc3_ir_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hst_wfc3_ir_f32     2 
$suncu -o $NCRES/mpc_comp_fspitzer_irac_f32    -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/spitzer_irac_f32    2 
$suncu -o $NCRES/mpc_comp_fg24_78_usb2_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/g24_78_usb2_f32     3 
$suncu -o $NCRES/mpc_comp_fjw_mirimage_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/jw_mirimage_f32     3 
$suncu -o $NCRES/mpc_comp_ftpch_order_f64      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/tpch_order_f64      1 
$suncu -o $NCRES/mpc_comp_ftpcxbb_store_f64    -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/tpcxbb_store_f64    2 
$suncu -o $NCRES/mpc_comp_ftpcxbb_web_f64      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/tpcxbb_web_f64      2 
$suncu -o $NCRES/mpc_comp_ftpch_lineitem_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpch_lineitem_f32   2 
$suncu -o $NCRES/mpc_comp_ftpcds_catalog_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpcds_catalog_f32   2 
$suncu -o $NCRES/mpc_comp_ftpcds_store_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpcds_store_f32     2 
$suncu -o $NCRES/mpc_comp_ftpcds_web_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpcds_web_f32       2 
# ================ #
# test decompress  #
# ================ #
$suncu -o $NCRES/mpc_decomp_fmsg_bt_f64          -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/msg_bt_f64.mpc          >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fnum_brain_f64       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/num_brain_f64.mpc       >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fnum_control_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/num_control_f64.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_frsim_f32            -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/rsim_f32.mpc            >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fastro_mhd_f64       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/astro_mhd_f64.mpc       >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fastro_pt_f64        -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/astro_pt_f64.mpc        >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fmiranda3d_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/miranda3d_f32.mpc       >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fturbulence_f32      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/turbulence_f32.mpc      >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fwave_f32            -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/wave_f32.mpc            >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fh3d_temp_f32        -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/h3d_temp_f32.mpc        >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fcitytemp_f32        -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/citytemp_f32.mpc        >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fts_gas_f32          -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/ts_gas_f32.mpc          >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fphone_gyro_f64      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/phone_gyro_f64.mpc      >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fwesad_chest_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/wesad_chest_f64.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fjane_street_f64     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/jane_street_f64.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fnyc_taxi2015_f64    -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/nyc_taxi2015_f64.mpc    >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fspain_gas_price_f64 -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/spain_gas_price_f64.mpc >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fsolar_wind_f32      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/solar_wind_f32.mpc      >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_facs_wht_f32         -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/acs_wht_f32.mpc         >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fhdr_night_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hdr_night_f32.mpc       >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fhdr_palermo_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hdr_palermo_f32.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fhst_wfc3_uvis_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hst_wfc3_uvis_f32.mpc   >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fhst_wfc3_ir_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/hst_wfc3_ir_f32.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fspitzer_irac_f32    -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/spitzer_irac_f32.mpc    >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fg24_78_usb2_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/g24_78_usb2_f32.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_fjw_mirimage_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/jw_mirimage_f32.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpch_order_f64      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/tpch_order_f64.mpc      >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpcxbb_store_f64    -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/tpcxbb_store_f64.mpc    >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpcxbb_web_f64      -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_double $DATADIR/tpcxbb_web_f64.mpc      >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpch_lineitem_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpch_lineitem_f32.mpc   >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpcds_catalog_f32   -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpcds_catalog_f32.mpc   >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpcds_store_f32     -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpcds_store_f32.mpc     >> $RESDIR/mpc_decomp.txt
$suncu -o $NCRES/mpc_decomp_ftpcds_web_f32       -f --set detailed --section SpeedOfLight_RooflineChart ./MPC_float  $DATADIR/tpcds_web_f32.mpc       >> $RESDIR/mpc_decomp.txt
rm $DATADIR/*.mpc $DATADIR/*.mpc.org