#!/bin/bash
EXECDIR=$WORKDIR/code/bitshuffle/tests
DATADIR=$WORKDIR/data
RESDIR=$WORKDIR/experiments
# ================ #
# test fpzip #
# ================ #
cd $EXECDIR
for j in {0..30} 
do
	echo $j "===================" 
	{ /usr/bin/time -v python f_33.py ;} 2>> /tmp/bitshuffle_mem.txt
	sed -i 's/K = '$j'/K = '$((j+1))'/' f_33.py
done
sed -i 's/K = 31/K = 0/' f_33.py
cat /tmp/bitshuffle_mem.txt | grep 'Maximum' | awk '{print $6}' >> $RESDIR/bitshuffle_all_mem.txt