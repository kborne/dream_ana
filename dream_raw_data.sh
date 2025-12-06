#!/bin/bash

source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

ABS_PATH="/sdf/data/lcls/ds/tmo/${EXPERIMENT}"

echo "Starting offline pre-processing for run ${RUN_NUM} in experiment ${EXPERIMENT}..."


sbatch --output="${ABS_PATH}/scratch/additional_metrics/logs/${RUN_NUM}_hexanode_hf_%j.log" \
       --account=lcls:$EXPERIMENT \
       --reservation=lcls:earlyscience \
       --partition=milano \
	   --nodes=1 \
	   --mem=0 \
       --wrap="python hexanode_health_elog.py -r \"$RUN_NUM\" -e \"$EXPERIMENT\""


echo "Job submitted for run ${RUN_NUM}. Check logs for progress."
