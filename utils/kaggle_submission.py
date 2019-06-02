import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from absl import flags

flags.DEFINE_integer("num_top_predicts", 20, "Number of Predictions")
FLAGS = flags.FLAGS

def inference(data_loader, model):
   '''  https://www.kaggle.com/artyomp/resnet50-baseline/code '''
   model.eval()

   activation = nn.Softmax(dim=1)
   all_predicts, all_confs, all_targets = [], [], []

   with torch.no_grad():
       for i, data in enumerate(tqdm(data_loader)):
           if data_loader.dataset.mode != 'test':
               input_, target = data
           else:
               input_, target = data, None

           output = model(input_.cuda())
           output = activation(output)

           confs, predicts = torch.topk(output, FLAGS.num_top_predicts)
           all_confs.append(confs)
           all_predicts.append(predicts)

           if target is not None:
               all_targets.append(target)

   predicts = torch.cat(all_predicts)
   confs = torch.cat(all_confs)
   targets = torch.cat(all_targets) if len(all_targets) else None

   return predicts, confs, targets


def generate_submission(test_loader, model, label_encoder):


 sample_sub = pd.read_csv('../input/landmark-recognition-2019/recognition_sample_submission.csv')

 predicts_gpu, confs_gpu, _ = inference(test_loader, model)
 predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

 labels = [label_encoder.inverse_transform(pred) for pred in predicts]
 print('labels')
 print(np.array(labels))
 print('confs')
 print(np.array(confs))

 sub = test_loader.dataset.dataframe
 def concat(label: np.ndarray, conf: np.ndarray) -> str:
     return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
 sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]

 sample_sub = sample_sub.set_index('id')
 sub = sub.set_index('id')
 sample_sub.update(sub)

 sample_sub.to_csv('submission.csv')

