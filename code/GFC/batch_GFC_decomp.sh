#!/bin/bash
DATADIR=/home/cc/data
OUTDIR=/home/cc/output
RESDIR=/home/cc/experiments
# ================ #
# test GFC #
# ================ #
{ ./GFC < $OUTDIR/msg_bt_f64.gfc          > $DATADIR/msg_bt_f64.gfcout            ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/num_brain_f64.gfc       > $DATADIR/num_brain_f64.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/num_control_f64.gfc     > $DATADIR/num_control_f64.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/rsim_f32.gfc            > $DATADIR/rsim_f32.gfcout              ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/astro_mhd_f64.gfc       > $DATADIR/astro_mhd_f64.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/astro_pt_f64.gfc        > $DATADIR/astro_pt_f64.gfcout          ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/miranda3d_f32.gfc       > $DATADIR/miranda3d_f32.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/turbulence_f32.gfc      > $DATADIR/turbulence_f32.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/wave_f32.gfc            > $DATADIR/wave_f32.gfcout              ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/h3d_temp_f32.gfc        > $DATADIR/h3d_temp_f32.gfcout          ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/citytemp_f32.gfc        > $DATADIR/citytemp_f32.gfcout          ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/ts_gas_f32.gfc          > $DATADIR/ts_gas_f32.gfcout            ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/phone_gyro_f64.gfc      > $DATADIR/phone_gyro_f64.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/wesad_chest_f64.gfc     > $DATADIR/wesad_chest_f64.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/jane_street_f64.gfc     > $DATADIR/jane_street_f64.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/nyc_taxi2015_f64.gfc    > $DATADIR/nyc_taxi2015_f64.gfcout      ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/spain_gas_price_f64.gfc > $DATADIR/spain_gas_price_f64.gfcout   ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/solar_wind_f32.gfc      > $DATADIR/solar_wind_f32.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/acs_wht_f32.gfc         > $DATADIR/acs_wht_f32.gfcout           ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/hdr_night_f32.gfc       > $DATADIR/hdr_night_f32.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/hdr_palermo_f32.gfc     > $DATADIR/hdr_palermo_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/hst_wfc3_uvis_f32.gfc   > $DATADIR/hst_wfc3_uvis_f32.gfcout     ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/hst_wfc3_ir_f32.gfc     > $DATADIR/hst_wfc3_ir_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/spitzer_irac_f32.gfc    > $DATADIR/spitzer_irac_f32.gfcout      ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/g24_78_usb2_f32.gfc     > $DATADIR/g24_78_usb2_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/jw_mirimage_f32.gfc     > $DATADIR/jw_mirimage_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpch_order_f64.gfc      > $DATADIR/tpch_order_f64.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/tpcxbb_store_f64.gfc    > $DATADIR/tpcxbb_store_f64.gfcout      ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/tpcxbb_web_f64.gfc      > $DATADIR/tpcxbb_web_f64.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
# { ./GFC < $OUTDIR/tpch_lineitem_f32.gfc   > $DATADIR/tpch_lineitem_f32.gfcout     ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpcds_catalog_f32.gfc   > $DATADIR/tpcds_catalog_f32.gfcout     ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpcds_store_f32.gfc     > $DATADIR/tpcds_store_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpcds_web_f32.gfc       > $DATADIR/tpcds_web_f32.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
rm $OUTDIR/*.gfc $DATADIR/*.gfcout
