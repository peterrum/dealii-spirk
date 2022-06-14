# Introduction/Motivation: GMG (Section 1)

```bash
mkdir gmg-scaling
cd gmg-scaling
python ../../IRK/experiments-skx/large-scaling-create-job-files-gmg.py 64
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

# Moderately parallel runs (Section 5.1)

```bash
mkdir -p small-scaling
cd small-scaling
python ../../IRK/scripts/small_scaling.py 4 16
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..
```

# Moderately parallel runs - key parameters (Section 5.1)

```bash
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
```

# Virtual topology and shared memory (Section 5.2)

```bash
mkdir -p parameters-vt
cd parameters-vt
python ../../IRK/scripts/parameters_vt.py
sbatch ../../IRK/experiments-skx/parameters-0048.cmd
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
sbatch ../../IRK/experiments-skx/parameters-3072.cmd
cd ..
```

# Large-scale parallel runs (Section 5.3)

```bash
mkdir -p large-scaling-2-1
cd large-scaling-2-1
python ../../IRK/scripts/large_scaling.py 2 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 2048
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-4-1
cd large-scaling-4-1
python ../../IRK/scripts/large_scaling.py 4 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 2048
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-9-1
cd large-scaling-9-1
python ../../IRK/scripts/large_scaling.py 9 1
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 2048
for cmd in *.cmd; do sbatch $cmd; done
cd ..



mkdir -p large-scaling-2-4
cd large-scaling-2-4
python ../../IRK/scripts/large_scaling.py 2 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 2048
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-4-4
cd large-scaling-4-4
python ../../IRK/scripts/large_scaling.py 4 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 2048
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-9-4
cd large-scaling-9-4
python ../../IRK/scripts/large_scaling.py 9 4
python ../../IRK/experiments-skx/large-scaling-create-job-files.py 2048
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

# Complex and batched runs (Section 5.4 + 6)

```bash
mkdir -p gmg-batched
cd gmg-batched
python ../../IRK/experiments-skx/large-scaling-create-job-files-gmg.py 64
sbatch node-0016.cmd
cd ..

mkdir -p parameters-complex-q
cd parameters-complex-q
python ../../IRK/scripts/parameters_q.py "complex_"
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-all-q
cd parameters-all-q
python ../../IRK/scripts/parameters_all_q.py
sbatch ../../IRK/experiments-skx/parameters-0768.cmd
cd ..
```