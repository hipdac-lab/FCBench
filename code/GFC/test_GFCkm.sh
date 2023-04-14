#!/bin/bash
DATADIR=../../../data/input
OUTDIR=../../../data/output
RESDIR=../../../data/experiments
# ================ #
# test GFC #
# ================ #
{ ./GFC 32 32 1 < $DATADIR/msg_bt_f64           > $OUTDIR/msg_bt_f64.gfc           ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 1 < $DATADIR/num_brain_f64        > $OUTDIR/num_brain_f64.gfc        ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 1 < $DATADIR/num_control_f64      > $OUTDIR/num_control_f64.gfc      ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/rsim_f32             > $OUTDIR/rsim_f32.gfc             ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 3 < $DATADIR/turbulence_f32       > $OUTDIR/turbulence_f32.gfc       ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 3 < $DATADIR/wave_f32             > $OUTDIR/wave_f32.gfc             ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 3 < $DATADIR/h3d_temp_f32         > $OUTDIR/h3d_temp_f32.gfc         ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 1 < $DATADIR/citytemp_f32         > $OUTDIR/citytemp_f32.gfc         ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 1 < $DATADIR/ts_gas_f32           > $OUTDIR/ts_gas_f32.gfc           ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/phone_gyro_f64       > $OUTDIR/phone_gyro_f64.gfc       ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/wesad_chest_f64      > $OUTDIR/wesad_chest_f64.gfc      ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/solar_wind_f32       > $OUTDIR/solar_wind_f32.gfc       ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/acs_wht_f32          > $OUTDIR/acs_wht_f32.gfc          ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/hdr_night_f32        > $OUTDIR/hdr_night_f32.gfc        ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/hst_wfc3_uvis_f32    > $OUTDIR/hst_wfc3_uvis_f32.gfc    ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/hst_wfc3_ir_f32      > $OUTDIR/hst_wfc3_ir_f32.gfc      ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/spitzer_irac_f32     > $OUTDIR/spitzer_irac_f32.gfc     ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 3 < $DATADIR/jw_mirimage_f32      > $OUTDIR/jw_mirimage_f32.gfc      ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 1 < $DATADIR/tpch_order_f64       > $OUTDIR/tpch_order_f64.gfc       ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/tpcds_catalog_f32    > $OUTDIR/tpcds_catalog_f32.gfc    ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/tpcds_store_f32      > $OUTDIR/tpcds_store_f32.gfc      ; } 2>> $RESDIR/gfc_comp.txt
{ ./GFC 32 32 2 < $DATADIR/tpcds_web_f32        > $OUTDIR/tpcds_web_f32.gfc        ; } 2>> $RESDIR/gfc_comp.txt
# ================
# test decompress
# ================
{ ./GFC < $OUTDIR/msg_bt_f64.gfc          > $DATADIR/msg_bt_f64.gfcout            ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/num_brain_f64.gfc       > $DATADIR/num_brain_f64.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/num_control_f64.gfc     > $DATADIR/num_control_f64.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/rsim_f32.gfc            > $DATADIR/rsim_f32.gfcout              ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/turbulence_f32.gfc      > $DATADIR/turbulence_f32.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/wave_f32.gfc            > $DATADIR/wave_f32.gfcout              ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/h3d_temp_f32.gfc        > $DATADIR/h3d_temp_f32.gfcout          ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/citytemp_f32.gfc        > $DATADIR/citytemp_f32.gfcout          ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/ts_gas_f32.gfc          > $DATADIR/ts_gas_f32.gfcout            ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/phone_gyro_f64.gfc      > $DATADIR/phone_gyro_f64.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/wesad_chest_f64.gfc     > $DATADIR/wesad_chest_f64.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/solar_wind_f32.gfc      > $DATADIR/solar_wind_f32.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/acs_wht_f32.gfc         > $DATADIR/acs_wht_f32.gfcout           ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/hdr_night_f32.gfc       > $DATADIR/hdr_night_f32.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/hst_wfc3_uvis_f32.gfc   > $DATADIR/hst_wfc3_uvis_f32.gfcout     ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/hst_wfc3_ir_f32.gfc     > $DATADIR/hst_wfc3_ir_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/spitzer_irac_f32.gfc    > $DATADIR/spitzer_irac_f32.gfcout      ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/jw_mirimage_f32.gfc     > $DATADIR/jw_mirimage_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpch_order_f64.gfc      > $DATADIR/tpch_order_f64.gfcout        ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpcds_catalog_f32.gfc   > $DATADIR/tpcds_catalog_f32.gfcout     ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpcds_store_f32.gfc     > $DATADIR/tpcds_store_f32.gfcout       ; } 2>> $RESDIR/gfc_decomp.txt
{ ./GFC < $OUTDIR/tpcds_web_f32.gfc       > $DATADIR/tpcds_web_f32.gfcout         ; } 2>> $RESDIR/gfc_decomp.txt
rm $OUTDIR/*.gfc $DATADIR/*.gfcout
