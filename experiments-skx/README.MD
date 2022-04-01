```bash
mkdir large-scaling-2-1
cd large-scaling-2-1
python ../../IRK/scripts/large_scaling.py 2 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 64
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir large-scaling-4-1
cd large-scaling-4-1
python ../../IRK/scripts/large_scaling.py 4 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 64
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir large-scaling-9-1
cd large-scaling-9-1
python ../../IRK/scripts/large_scaling.py 9 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 64
for cmd in *.cmd; do sbatch $cmd; done
cd ..



mkdir large-scaling-2-4
cd large-scaling-2-4
python ../../IRK/scripts/large_scaling.py 2 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir large-scaling-4-4
cd large-scaling-4-4
python ../../IRK/scripts/large_scaling.py 4 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir large-scaling-9-4
cd large-scaling-9-4
python ../../IRK/scripts/large_scaling.py 9 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

```