# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi
#export PATH="/opt/anaconda3.6/bin:$PATH"
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
#export PATH=/usr/local/cuda-8.0/bin:$PATH
#export CUDA_HOME=/usr/local/cuda-8.0
#export CFLAGS='-Wall -Wextra -std=c99'
# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
export PATH="/opt/anaconda3.6/bin:/opt/anaconda3/bin:~/.local/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda-9.0/lib64/:/opt/cudnn-7.0/lib/:$LD_LIBRARY_PATH"
