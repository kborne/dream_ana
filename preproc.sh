#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --output="/sdf/data/lcls/ds/tmo/"$1"/scratch/xiangli/log/"$2_%j".out"
#SBATCH --error="/sdf/data/lcls/ds/tmo/"$1"/scratch/xiangli/log/"$2_%j".err"
#SBATCH --partition=milano
#SBATCH --account=lcls:tmo101347825
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --exclusive
#SBATCH --mail-user=xiangli@slac.stanford.edu
#SBATCH --mail-type=FAIL,END
mpirun python -u -m mpi4py.run $(which dream) --exp=$1 --run=$2
EOT

