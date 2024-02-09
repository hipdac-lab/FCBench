#!/bin/bash
## ========================= #
## setup directories
## ========================= #
RESDIR=$WORKDIR/experiments
# =========== #
# test pFPC   #
# =========== #
awk '{ print $2 }' $RESDIR/pfpc_comp.txt > /tmp/pfpc_res.txt
awk '{ print $7 }' $RESDIR/pfpc_comp.txt >> /tmp/pfpc_res.txt
awk '{ print $5 }' $RESDIR/pfpc_decomp.txt >> /tmp/pfpc_res.txt

# =========== #
# test SPDP   #
# =========== #
awk '{ print $2 }' $RESDIR/spdp_comp.txt > /tmp/spdp_res.txt
awk '{ print $7 }' $RESDIR/spdp_comp.txt >> /tmp/spdp_res.txt
awk '{ print $5 }' $RESDIR/spdp_decomp.txt >> /tmp/spdp_res.txt

# ============== #
# test fpzip #
# ============== #
sed 's/\=/\ /g' $RESDIR/fpzip_all.txt | awk '{ print $4, $6, $9 }' > /tmp/fpzip_res.txt

## ================ #
## test bitshuffle
## ================ #
awk '{ print $4, $6}' $RESDIR/bitshuffle_lz.txt > /tmp/shflz_res.txt
awk '{ print $4, $6}' $RESDIR/bitshuffle_zstd.txt > /tmp/shfzstd_res.txt

# ================ #
# test ndzip-CPU   #
# ================ #
sed 's/size/0 0 0 0 0 0 0 0 0 0 0 0 0 0/g' $RESDIR/ndzc_comp.txt | awk '{ print $11 }' | sed 's/\,//g'  > /tmp/ndzc_res.txt
sed 's/size/0 0 0 0 0 0 0 0 0 0 0 0 0 0/g' $RESDIR/ndzc_comp.txt | awk '{ print $15 }' | sed 's/\,//g' >> /tmp/ndzc_res.txt
awk '{ print $5 }' $RESDIR/ndzc_decomp.txt | sed 's/s\,//g' >> /tmp/ndzc_res.txt
# =========== #
# test BUFF   #
# =========== #
grep 'Performance:' $RESDIR/buff_all.txt | awk '{ print $4, $5, $6 }' > /tmp/buff_res.txt

# ============== #
# test gorilla #
# ============== #
grep 'cr:' $RESDIR/gorilla_all.txt | sed 's/\:/\ /g'  | awk '{ print $5, $7, $9 }' > /tmp/gorilla_res.txt

# ============== #
# test chimp #
# ============== #
grep 'cr:' $RESDIR/chimp_all.txt | sed 's/\:/\ /g'  | awk '{ print $5, $7, $9 }' > /tmp/chimp_res.txt

# ========== #
# test GFC   #
# ========== #
awk '{ print $2 }' $RESDIR/gfc_comp.txt > /tmp/gfc_res.txt
awk '{ print $10 }' $RESDIR/gfc_comp.txt >> /tmp/gfc_res.txt
awk '{ print $10 }' $RESDIR/gfc_decomp.txt >> /tmp/gfc_res.txt

# ========== #
# test MPC   #
# ========== #
awk '{ print $8 }' $RESDIR/mpc_comp.txt > /tmp/mpc_res.txt
awk '{ print $5 }' $RESDIR/mpc_comp.txt >> /tmp/mpc_res.txt
awk '{ print $5 }' $RESDIR/mpc_decomp.txt >> /tmp/mpc_res.txt

# =============================== #
# test nvcomp:LZ4 and nvcomp:zstd #
# =============================== #
grep 'compressed ratio:' $RESDIR/nvlz4.txt | awk '{ print $5 }' > /tmp/nvlz4_res.txt
grep 'compression throughput' $RESDIR/nvlz4.txt | sed -n 'p;n' | awk '{ print $4 }' >> /tmp/nvlz4_res.txt
grep 'compression throughput' $RESDIR/nvlz4.txt | sed -n 'n;p' | awk '{ print $4 }' >> /tmp/nvlz4_res.txt
grep 'compressed ratio:' $RESDIR/nvbitcomp.txt | awk '{ print $5 }' > /tmp/nvbitcomp_res.txt
grep 'compression throughput' $RESDIR/nvbitcomp.txt | sed -n 'p;n' | awk '{ print $4 }' >> /tmp/nvbitcomp_res.txt
grep 'compression throughput' $RESDIR/nvbitcomp.txt | sed -n 'n;p' | awk '{ print $4 }' >> /tmp/nvbitcomp_res.txt

# ================ #
# test ndzip-GPU   #
# ================ #
sed 's/size/0 0 0 0 0 0 0 0 0 0 0 0 0 0/g' $RESDIR/ndzg_comp.txt | awk '{ print $11 }' | sed 's/\,//g'  > /tmp/ndzg_res.txt
sed 's/size/0 0 0 0 0 0 0 0 0 0 0 0 0 0/g' $RESDIR/ndzg_comp.txt | awk '{ print $15 }' | sed 's/\,//g' >> /tmp/ndzg_res.txt
awk '{ print $5 }' $RESDIR/ndzg_decomp.txt | sed 's/s\,//g' >> /tmp/ndzg_res.txt
