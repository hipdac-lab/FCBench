#!/bin/bash
## ========================= #
## setup directories
## ========================= #
DATADIR=$MYDATA
BASEDIR=$MYSCRATCH/code
RESDIR=$MYSCRATCH/experiments
=========== #
test pFPC   #
=========== #
awk '{ print $2 }' $RESDIR/pfpc_comp.txt > /tmp/pfpc_res.txt
awk '{ print $7 }' $RESDIR/pfpc_comp.txt >> /tmp/pfpc_res.txt
awk '{ print $4 }' $RESDIR/pfpc_decomp.txt >> /tmp/pfpc_res.txt

=========== #
test SPDP   #
=========== #
awk '{ print $2 }' $RESDIR/spdp_comp.txt > /tmp/spdp_res.txt
awk '{ print $7 }' $RESDIR/spdp_comp.txt >> /tmp/spdp_res.txt
awk '{ print $5 }' $RESDIR/spdp_decomp.txt >> /tmp/spdp_res.txt

============== #
test fpzip #
============== #
sed 's/\=/\ /g' $RESDIR/fpzip_all.txt | awk '{ print $4, $6, $9 }' > /tmp/fpzip_res.txt

## ================ #
## test bitshuffle
## ================ #
awk '{ print $4, $6}' $RESDIR/bitshuffle_lz.txt > /tmp/shflz_res.txt
awk '{ print $4, $6}' $RESDIR/bitshuffle_zstd.txt > /tmp/shfzstd_res.txt

================ #
test ndzip-CPU   #
================ #
awk '{ print $11, $14 }' $RESDIR/ndzc_comp.txt | sed 's/s\,//g' | sed 's/\,//g' > /tmp/ndzc_res.txt
awk '{ print $4 }' $RESDIR/ndzc_decomp.txt | sed 's/s\,//g' >> /tmp/ndzc_res.txt

# =========== #
# test BUFF   #
# =========== #
for i in ${nvcompn[*]}; do
	printf 'fn[%s]=%20s\t (%sD)\n' "$i" "${fn[i]}" "${dm[i]}"
	$BASEDIR/buff/database/target/release/comp_profiler $DATADIR/${fn[i]} buff-simd64 100000000 1.15 >> $RESDIR/buff_all.txt
done

============== #
test gorilla #
============== #
cd $BASEDIR/influxdb
git checkout gorilla
go clean -testcache
go test  -test.timeout 0 -run TestCompress_XC2 -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 >> $RESDIR/gorilla_all.txt

============== #
test chimp #
============== #
RESDIR=$MYSCRATCH/experiments
cd $BASEDIR/influxdb
git checkout chimp128
go clean -testcache
go test  -test.timeout 0 -run TestCompress_XC2 -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 >> $RESDIR/chimp_all.txt


# ========== #
# test GFC   #
# ========== #
for i in ${gfcn[*]}; do
	{ $BASEDIR/GFC/GFC 32 32 ${dm[i]} < $DATADIR/${fn[i]}  > $OUTDIR/${fn[i]}.gfc ; } 2>> $RESDIR/gfc_comp.txt
	{ $BASEDIR/GFC/GFC < $OUTDIR/${fn[i]}.gfc  > $OUTDIR/${fn[i]}.gfcout ; } 2>> $RESDIR/gfc_decomp.txt
done

# ========== #
# test MPC   #
# ========== #
for i in "${!fn[@]}"; do
	p="${mpcn[i]}"
	printf 'fn[%s]=%20s\t exec=%s\t (%sD)\n' "$i" "${fn[i]}" "${mpcp[$p]}" "${dm[i]}"
	$BASEDIR/MPC/${mpcp[$p]} $DATADIR/${fn[i]}  ${dm[i]} >> $RESDIR/mpc_comp.txt
	$BASEDIR/MPC/${mpcp[$p]} $DATADIR/${fn[i]}.mpc       >> $RESDIR/mpc_decomp.txt
done

# =============================== #
# test nvcomp:LZ4 and nvcomp:zstd #
# =============================== #
for i in "${!fn[@]}"; do
	printf 'fn[%s]=%20s\t (%sD)\n' "$i" "${fn[i]}" "${dm[i]}"
	$BASEDIR/nvbench/bin/benchmark_lz4_chunked -f $DATADIR/${fn[i]} >> $RESDIR/nvlz4.txt
	$BASEDIR/nvbench/bin/benchmark_bitcomp_chunked -f $DATADIR/${fn[i]}  >> $RESDIR/nvbitcomp.txt
done

================ #
test ndzip-GPU   #
================ #
awk '{ print $11, $14 }' $RESDIR/ndzg_comp.txt | sed 's/s\,//g' | sed 's/\,//g' > /tmp/ndzg_res.txt
awk '{ print $4 }' $RESDIR/ndzg_decomp.txt | sed 's/s\,//g' >> /tmp/ndzg_res.txt
