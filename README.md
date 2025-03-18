On slurm, create conda environment
```
source ~/.bashrc

conda create -n bni python=3.10

git clone https://github.com/TCQian/comp-resist-perturb.git

cd comp-resist-perturb

pip install -r requirements.txt
```

Once done setting up, just run the bash script:
```
sbatch slurm_sd_train.sh
```