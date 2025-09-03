benchmark="eval_plus"
conda create -n eval_${benchmark}_env python=3.9
source /home/tiger/miniconda3/bin/activate
conda activate eval_${benchmark}_env
cd ${benchmark}
pip install -r requirements.txt