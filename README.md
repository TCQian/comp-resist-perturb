On slurm, create conda environment
```
source ~/.bashrc

conda create -n bni python=3.10
```

Run the bash script:
```
sbatch slurm_sd_train.sh
```