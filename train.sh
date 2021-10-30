# Pretrain and finetune
for job_ix in {1..6}
do
    echo "=================================================================="
    echo "Start pretrain for job$job_ix"
    cd job$job_ix
    python3 -u pretrain.py
    echo "=================================================================="
    echo "Start finetune for job$job_ix"
    python3 -u finetune.py
    cd ../
done
echo "Finish all"
echo "=================================================================="
# Start ensemble
echo "=================================================================="
echo "Start ensemble"
python3 -u ensemble.py
echo "Finish all"
echo "=================================================================="
