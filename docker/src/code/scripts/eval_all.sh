#!/bin/bash
## ========================= #
## setup directories
## ========================= #
export DATADIR=$WORKDIR/data
BASEDIR=$WORKDIR/code
RESDIR=$WORKDIR/experiments
OUTDIR=$WORKDIR/output
if [ ! -d $RESDIR ]
then
	mkdir $RESDIR
fi

if [ ! -d $OUTDIR ]
then
	mkdir $OUTDIR
fi
. testfiles.cfg
## ================ #
## test bitshuffle
## ================ #
# FACTOR=1 ## ... defaut BLOCK = 1024 elements
# source $HOME/env4shuffle/bin/activate
# cd $WORKDIR/code/bitshuffle/tests
# for j in {0..30}
# do
# 	python f_33_lz.py $FACTOR $j 	>> $RESDIR/bitshuffle_lz.txt
# 	python f_33_zstd.py $FACTOR $j 	>> $RESDIR/bitshuffle_zstd.txt
# done

# # ========== #
# # test GFC   #
# # ========== #
for i in ${gfcn[*]}; do
	{ $BASEDIR/GFC/GFC 32 32 ${dm[i]} < $DATADIR/${fn[i]}  > $OUTDIR/${fn[i]}.gfc ; } 2>> $RESDIR/gfc_comp.txt
	{ $BASEDIR/GFC/GFC < $OUTDIR/${fn[i]}.gfc  > $OUTDIR/${fn[i]}.gfcout ; } 2>> $RESDIR/gfc_decomp.txt
done

# # ========== #
# # test MPC   #
# # ========== #
# for i in "${!fn[@]}"; do
# 	p="${mpcn[i]}"
# 	printf 'fn[%s]=%20s\t exec=%s\t (%sD)\n' "$i" "${fn[i]}" "${mpcp[$p]}" "${dm[i]}"
# 	$BASEDIR/MPC/${mpcp[$p]} $DATADIR/${fn[i]}  ${dm[i]} >> $RESDIR/mpc_comp.txt
# 	$BASEDIR/MPC/${mpcp[$p]} $DATADIR/${fn[i]}.mpc       >> $RESDIR/mpc_decomp.txt
# done

# # =============================== #
# # test nvcomp:LZ4 and nvcomp:zstd #
# # =============================== #
# for i in "${!fn[@]}"; do
# 	printf 'fn[%s]=%20s\t (%sD)\n' "$i" "${fn[i]}" "${dm[i]}"
# 	$BASEDIR/nvbench/bin/benchmark_lz4_chunked -f $DATADIR/${fn[i]} >> $RESDIR/nvlz4.txt
# 	$BASEDIR/nvbench/bin/benchmark_bitcomp_chunked -f $DATADIR/${fn[i]}  >> $RESDIR/nvbitcomp.txt
# done

# # =========== #
# # test BUFF   #
# # =========== #
# for i in ${nvcompn[*]}; do
# 	printf 'fn[%s]=%20s\t (%sD)\n' "$i" "${fn[i]}" "${dm[i]}"
# 	$BASEDIR/buff/database/target/release/comp_profiler $DATADIR/${fn[i]} buff-simd64 100000000 1.15 >> $RESDIR/buff_all.txt
# done

# ============== #
# test fpzip #
# ============== #
# for i in ${fpzipn[*]}; do
# 	{ $BASEDIR/fpzip/build/bin/fpzip -i $DATADIR/${fn[i]}  -t ${dt[i]} -${dm[i]}  ${shape[i]} ; } 2>> $RESDIR/fpzip_all.txt
# done

# ============== #
# test chimp #
# ============== #
# RESDIR=$WORKDIR/experiments
# cd $BASEDIR/influxdb
# git checkout chimp128
# go clean -testcache
# go test  -test.timeout 0 -run TestCompress_XC2 -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 >> $RESDIR/chimp_all.txt

# ============== #
# test gorilla #
# ============== #
# cd $BASEDIR/influxdb
# git checkout gorilla
# go clean -testcache
# go test  -test.timeout 0 -run TestCompress_XC2 -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 >> $RESDIR/gorilla_all.txt

# =========== #
# test pFPC   #
# =========== #
# SIZE=1048576 
# for i in "${!fn[@]}"; do
# 	printf 'fn[%s]=%20s\t (%sD)\n' "$i" "${fn[i]}" "${dm[i]}"
# 	# { $BASEDIR/pFPC/pfpcb 20 8 2048 $SIZE < $DATADIR/${fn[i]}  > $OUTDIR/${fn[i]}.pfpc ;} 2>> $RESDIR/pfpc_comp.txt
# 	{ $BASEDIR/pFPC/pfpcb $SIZE < $OUTDIR/${fn[i]}.pfpc  > $OUTDIR/${fn[i]}.pfpcout ;} 2>> $RESDIR/pfpc_decomp.txt
# done

# =========== #
# test SPDP   #
# =========== #
# SIZE=1048576 
# for i in "${!fn[@]}"; do
# 	printf 'fn[%s]=%20s\t (%sD)\n' "$i" "${fn[i]}" "${dm[i]}"
# 	{ $BASEDIR/SPDP/spdpb 10 $SIZE < $DATADIR/${fn[i]}  > $OUTDIR/${fn[i]}.spdp ;} 2>> $RESDIR/spdp_comp.txt
# 	{ $BASEDIR/SPDP/spdpb $SIZE < $OUTDIR/${fn[i]}.spdp  > $DATADIR/${fn[i]}.spdpout ;} 2>> $RESDIR/spdp_decomp.txt
# done

# ================ #
# test ndzip-CPU   #
# ================ #
# for i in ${ndzcpu[*]}; do
# 	{ $BASEDIR/ndzip/build/compress -i $DATADIR/${fn[i]}  -o $OUTDIR/${fn[i]}.ndzc -t ${dt[i]} -n ${shape[i]} ; } 2>> $RESDIR/ndzc_comp.txt
# 	{ $BASEDIR/ndzip/build/compress -d -i $OUTDIR/${fn[i]}.ndzc  -o $OUTDIR/${fn[i]}.ndzcout -t ${dt[i]} -n ${shape[i]} ; } 2>> $RESDIR/ndzc_decomp.txt
# done

# ================ #
# test ndzip-GPU   #
# ================ #
# for i in ${ndzgpu[*]}; do
# 	{ $BASEDIR/ndzip/build/compress -e cuda -i $DATADIR/${fn[i]}  -o $OUTDIR/${fn[i]}.ndzc -t ${dt[i]} -n ${shape[i]} ; } 2>> $RESDIR/ndzg_comp.txt
# 	{ $BASEDIR/ndzip/build/compress -e cuda -d -i $OUTDIR/${fn[i]}.ndzc  -o $OUTDIR/${fn[i]}.ndzcout -t ${dt[i]} -n ${shape[i]} ; } 2>> $RESDIR/ndzg_decomp.txt
# done