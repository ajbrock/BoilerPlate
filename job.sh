#!/bin/bash
# export HOME=/afs/inf.ed.ac.uk/user/s15/s1580274/
# export PATH="/afs/inf.ed.ac.uk/user/s15/s1580274/user/bin/:/afs/inf.ed.ac.uk/group/project/modelnet/cuda_7.5/bin/:$PATH"
# export LD_LIBRARY_PATH=/afs/inf.ed.ac.uk/group/project/modelnet/cuda_7.5/lib64/:/mylibc6/ld-linux-x86-64.so.2/:/afs/inf.ed.ac.uk/user/s15/s1580274/lib64/
# export CPATH=/afs/inf.ed.ac.uk/user/s15/s1580274/lib64/include/
# export KRB5CCNAME=s1580274
# export uid=s1580274

# export CUDA_HOME=/afs/inf.ed.ac.uk/group/project/modelnet/cuda_7.5/
export CUDA_HOME=/opt/cuda-8.0.44
export CUDNN_HOME=/home/s1580274/cudnn6
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:/home/s1580274/:/home/s1580274/miniconda3/bin/:${PATH}


export PYTHON_PATH=$PATH
# export stdout_path=/home/s1580274
# Activate the relevant virtual environment:
# /home/s1580274/miniconda3/envs/ajenv
source activate ajenv
# source /<path_to_virtual_environment>/activate
 

# MAKE SURE TO USE THIS, WHATEVER ENVIRONMENT YOU ARE USING.
# The path should point to the location where you saved the gpu_lock_script provided below.
# It's important to include this, as it prevents collisions on the GPUs.
source gpu_lock_script.sh

 
## get a kerberos ticket


# Everything above this point is applicable to all CUDA based packages (Theano, tensorflow, etc).
# The things below might change depending on the environment.
 

python train.py --test --no-validate --epochs=100 --validate-every=5 --model=SERN
