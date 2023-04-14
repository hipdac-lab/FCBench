#!/bin/bash
RESDIR=/home/cc/experiments
for i in {1..9}
do
	echo "#" $i "================"	>> $RESDIR/buff_all.txt
	bash test_buff_p10.sh
done