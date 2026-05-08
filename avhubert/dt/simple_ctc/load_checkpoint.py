import fairseq
import sys
sys.path.append('../..')
# # 添加项目根目录到 sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)
import hubert_pretraining, hubert

import ipdb

ckpt_path = "/hdd2/avhubert_exp/pretrain/clean_pt_cncvs3_speech_1500000_iter2/checkpoints/checkpoint_best.pt"

models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
ipdb.set_trace()

print()
