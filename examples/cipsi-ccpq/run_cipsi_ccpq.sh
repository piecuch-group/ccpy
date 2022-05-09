#!/bin/bash

notify_email ()
{
	date=$(date)
	content="
CIPSI-driven CC(P;Q) using 2BA is done.
Finished on $date on $HOSTNAME.
Working directory $(pwd).
"
	echo "$content" | mail -s 'CIPSI-driven CC(P;Q) Job Done' gururang@msu.edu
}


#STORAGE_DIR="/storage/gururang/for_storage_from_coupled/calculations/f2-pvdz-gms/f2-1.0-v3/ndet_"
STORAGE_DIR="/storage/gururang/for_storage_from_coupled/calculations/f2-pvdz-gms/f2-1.5-v3/ndet_"
#STORAGE_DIR="/storage/gururang/for_storage_from_coupled/calculations/f2-pvdz-gms/f2-2.0-v3/ndet_"
#STORAGE_DIR="/storage/gururang/for_storage_from_coupled/calculations/f2-pvdz-gms/f2-5.0-v3/ndet_"

CI_FILE_DIR="/scratch/gururang/test_ccpq_2ba/"
N_DET=(10 100 1000 5000 10000 50000 100000 500000 1000000)

#mkdir ${1}
#mkdir ${CI_FILE_DIR}${1}

for n in ${N_DET[@]}; do
    FILE_LOC=${STORAGE_DIR}${n}"/ci.vectors.dat"
    OUTPUT_FILE=${CI_FILE_DIR}${1}"/civecs-"${n}".dat"

    echo "Processing CI vector file from "${FILE_LOC}
    python /home2/gururang/code/cipsi_civec_parse.py -f 4 -o ${OUTPUT_FILE} ${FILE_LOC}

    echo "Running CC(P;Q) calculation"
    python -u f2_ccpq_2ba.py "-civecs" ${OUTPUT_FILE} > ${1}"/f2-"${n}"-2ba.out"

done

echo 'All CC(P;Q) calculations are completed'
notify_email
