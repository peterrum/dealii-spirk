```bash
mkdir -p large-scaling-2-1
cd large-scaling-2-1
python ../../IRK/scripts/large_scaling.py 2 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-4-1
cd large-scaling-4-1
python ../../IRK/scripts/large_scaling.py 4 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-9-1
cd large-scaling-9-1
python ../../IRK/scripts/large_scaling.py 9 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..



mkdir -p large-scaling-2-4
cd large-scaling-2-4
python ../../IRK/scripts/large_scaling.py 2 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-4-4
cd large-scaling-4-4
python ../../IRK/scripts/large_scaling.py 4 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-9-4
cd large-scaling-9-4
python ../../IRK/scripts/large_scaling.py 9 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

```bash
mkdir -p parameters-vt
cd parameters-vt
python ../../IRK/scripts/parameters_vt.py
sbatch ../../IRK/experiments-skx/parameters-0048.cmd
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
sbatch ../../IRK/experiments-skx/parameters-3072.cmd
cd ..

mkdir -p parameters-dt
cd parameters-dt
python ../../IRK/scripts/parameters_dt.py
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-p
cd parameters-p
python ../../IRK/scripts/parameters_p.py
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-q
cd parameters-q
python ../../IRK/scripts/parameters_q.py
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-tolerance
cd parameters-tolerance
python ../../IRK/scripts/parameters_tolerance.py
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..
```

```bash
mkdir gmg-scaling
cd gmg-scaling
python ../../IRK/experiments-skx/large-scaling-create-job-files-gmg.py 64
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

```bash
mkdir -p small-scaling
cd small-scaling
python ../../IRK/scripts/small_scaling.py 4 16
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..
```


```bash
mkdir -p parameters-complex-q
cd parameters-complex-q
python ../../IRK/scripts/parameters_q.py "complex_"
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..
```