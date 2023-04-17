# ============== #
# test ndzip #
# ============== #
EXECDIR=/home/cc/code
DATADIR=/home/cc/data
OUTDIR=/home/cc/output
RESDIR=/home/cc/experiments
# ================ #
# test ndzip-gpu #
# ================ #
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/msg_bt_f64          -o $OUTDIR/msg_bt_f64.ndzip    		 -t double -n 33298679       ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/num_brain_f64       -o $OUTDIR/num_brain_f64.ndzip       -t double -n 17730000       ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/num_control_f64     -o $OUTDIR/num_control_f64.ndzip     -t double -n 19938093       ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/rsim_f32            -o $OUTDIR/rsim_f32.ndzip            -t float  -n 2048 11509     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/astro_mhd_f64       -o $OUTDIR/astro_mhd_f64.ndzip       -t double -n 130 514 1026   ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/astro_pt_f64        -o $OUTDIR/astro_pt_f64.ndzip        -t double -n 512 256 640    ; } 2>>$RESDIR/ndzip_comp_gpu.txt
# { $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/miranda3d_f32       -o $OUTDIR/miranda3d_f32.ndzip       -t float  -n 1024 1024 1024 ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/turbulence_f32      -o $OUTDIR/turbulence_f32.ndzip      -t float  -n 256 256 256    ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/wave_f32            -o $OUTDIR/wave_f32.ndzip            -t float  -n 512 512 512    ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/h3d_temp_f32        -o $OUTDIR/h3d_temp_f32.ndzip        -t float  -n 100 500 500    ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/citytemp_f32        -o $OUTDIR/citytemp_f32.ndzip        -t float  -n 2906326        ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/ts_gas_f32          -o $OUTDIR/ts_gas_f32.ndzip          -t float  -n 76863200       ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/phone_gyro_f64      -o $OUTDIR/phone_gyro_f64.ndzip      -t double -n 13932632 3     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/wesad_chest_f64     -o $OUTDIR/wesad_chest_f64.ndzip     -t double -n 4255300 8      ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/jane_street_f64     -o $OUTDIR/jane_street_f64.ndzip     -t double -n 1664520 136    ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/nyc_taxi2015_f64    -o $OUTDIR/nyc_taxi2015_f64.ndzip    -t double -n 12748986 7     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/spain_gas_price_f64 -o $OUTDIR/spain_gas_price_f64.ndzip -t double -n 36942486 3     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/solar_wind_f32      -o $OUTDIR/solar_wind_f32.ndzip      -t float  -n 7571081 14     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/acs_wht_f32         -o $OUTDIR/acs_wht_f32.ndzip         -t float  -n 7500 7500      ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/hdr_night_f32       -o $OUTDIR/hdr_night_f32.ndzip       -t float  -n 8192 16384     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/hdr_palermo_f32     -o $OUTDIR/hdr_palermo_f32.ndzip     -t float  -n 10268 20536    ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/hst_wfc3_uvis_f32   -o $OUTDIR/hst_wfc3_uvis_f32.ndzip   -t float  -n 5329 5110      ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/hst_wfc3_ir_f32     -o $OUTDIR/hst_wfc3_ir_f32.ndzip     -t float  -n 2484 2417      ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/spitzer_irac_f32    -o $OUTDIR/spitzer_irac_f32.ndzip    -t float  -n 6456 6389      ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/g24_78_usb2_f32     -o $OUTDIR/g24_78_usb2_f32.ndzip     -t float  -n 2426 371 371   ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/jw_mirimage_f32     -o $OUTDIR/jw_mirimage_f32.ndzip     -t float  -n 40 1024 1032   ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpch_order_f64      -o $OUTDIR/tpch_order_f64.ndzip      -t double -n 15000000       ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpcxbb_store_f64    -o $OUTDIR/tpcxbb_store_f64.ndzip    -t double -n 8228343 12     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpcxbb_web_f64      -o $OUTDIR/tpcxbb_web_f64.ndzip      -t double -n 8223189 15     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpch_lineitem_f32   -o $OUTDIR/tpch_lineitem_f32.ndzip   -t float  -n 59986051 4     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpcds_catalog_f32   -o $OUTDIR/tpcds_catalog_f32.ndzip   -t float  -n 2880058 15     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpcds_store_f32     -o $OUTDIR/tpcds_store_f32.ndzip     -t float  -n 5760749 12     ; } 2>>$RESDIR/ndzip_comp_gpu.txt
{ $EXECDIR/ndzip/build/compress -e cuda -i $DATADIR/tpcds_web_f32       -o $OUTDIR/tpcds_web_f32.ndzip       -t float  -n 1439247 15     ; } 2>>$RESDIR/ndzip_comp_gpu.txt

echo "test ndzip-gpu decompress"
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/msg_bt_f64.ndzip    		-o $DATADIR/msg_bt_f64.ndzout          -t double -n 33298679       ; } 2>>$RESDIR/ndzip_decomp_gpu.txt 
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/num_brain_f64.ndzip       -o $DATADIR/num_brain_f64.ndzout       -t double -n 17730000       ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/num_control_f64.ndzip     -o $DATADIR/num_control_f64.ndzout     -t double -n 19938093       ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/rsim_f32.ndzip            -o $DATADIR/rsim_f32.ndzout            -t float  -n 2048 11509     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/astro_mhd_f64.ndzip       -o $DATADIR/astro_mhd_f64.ndzout       -t double -n 130 514 1026   ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/astro_pt_f64.ndzip        -o $DATADIR/astro_pt_f64.ndzout        -t double -n 512 256 640    ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
# { $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/miranda3d_f32.ndzip       -o $DATADIR/miranda3d_f32.ndzout       -t float  -n 1024 1024 1024 ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/turbulence_f32.ndzip      -o $DATADIR/turbulence_f32.ndzout      -t float  -n 256 256 256    ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/wave_f32.ndzip            -o $DATADIR/wave_f32.ndzout            -t float  -n 512 512 512    ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/h3d_temp_f32.ndzip        -o $DATADIR/h3d_temp_f32.ndzout        -t float  -n 100 500 500    ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/citytemp_f32.ndzip        -o $DATADIR/citytemp_f32.ndzout        -t float  -n 2906326        ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/ts_gas_f32.ndzip          -o $DATADIR/ts_gas_f32.ndzout          -t float  -n 76863200       ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/phone_gyro_f64.ndzip      -o $DATADIR/phone_gyro_f64.ndzout      -t double -n 13932632 3     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/wesad_chest_f64.ndzip     -o $DATADIR/wesad_chest_f64.ndzout     -t double -n 4255300 8      ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/jane_street_f64.ndzip     -o $DATADIR/jane_street_f64.ndzout     -t double -n 1664520 136    ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/nyc_taxi2015_f64.ndzip    -o $DATADIR/nyc_taxi2015_f64.ndzout    -t double -n 12748986 7     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/spain_gas_price_f64.ndzip -o $DATADIR/spain_gas_price_f64.ndzout -t double -n 36942486 3     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/solar_wind_f32.ndzip      -o $DATADIR/solar_wind_f32.ndzout      -t float  -n 7571081 14     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/acs_wht_f32.ndzip         -o $DATADIR/acs_wht_f32.ndzout         -t float  -n 7500 7500      ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/hdr_night_f32.ndzip       -o $DATADIR/hdr_night_f32.ndzout       -t float  -n 8192 16384     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/hdr_palermo_f32.ndzip     -o $DATADIR/hdr_palermo_f32.ndzout     -t float  -n 10268 20536    ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/hst_wfc3_uvis_f32.ndzip   -o $DATADIR/hst_wfc3_uvis_f32.ndzout   -t float  -n 5329 5110      ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/hst_wfc3_ir_f32.ndzip     -o $DATADIR/hst_wfc3_ir_f32.ndzout     -t float  -n 2484 2417      ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/spitzer_irac_f32.ndzip    -o $DATADIR/spitzer_irac_f32.ndzout    -t float  -n 6456 6389      ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/g24_78_usb2_f32.ndzip     -o $DATADIR/g24_78_usb2_f32.ndzout     -t float  -n 2426 371 371   ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/jw_mirimage_f32.ndzip     -o $DATADIR/jw_mirimage_f32.ndzout     -t float  -n 40 1024 1032   ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpch_order_f64.ndzip      -o $DATADIR/tpch_order_f64.ndzout      -t double -n 15000000       ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpcxbb_store_f64.ndzip    -o $DATADIR/tpcxbb_store_f64.ndzout    -t double -n 8228343 12     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpcxbb_web_f64.ndzip      -o $DATADIR/tpcxbb_web_f64.ndzout      -t double -n 8223189 15     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpch_lineitem_f32.ndzip   -o $DATADIR/tpch_lineitem_f32.ndzout   -t float  -n 59986051 4     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpcds_catalog_f32.ndzip   -o $DATADIR/tpcds_catalog_f32.ndzout   -t float  -n 2880058 15     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpcds_store_f32.ndzip     -o $DATADIR/tpcds_store_f32.ndzout     -t float  -n 5760749 12     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
{ $EXECDIR/ndzip/build/compress -d -e cuda -i $OUTDIR/tpcds_web_f32.ndzip       -o $DATADIR/tpcds_web_f32.ndzout       -t float  -n 1439247 15     ; } 2>>$RESDIR/ndzip_decomp_gpu.txt
rm $OUTDIR/*.ndzip $DATADIR/*.ndzout

