
import os
from experiment import Experiment

from analyzer import Analyzer
import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
  
  urg_levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
  repn=10000
  exp_maxt=15000
  
  coh_list=[0,0.016,0.032,0.064,0.128]


  from utils import init_with_args
  model,setting,eval_only=init_with_args()
  exp=Experiment(model=model,setting=setting,repn=repn,exp_maxt=exp_maxt,coh_list=coh_list)


  base_A_urg=exp.A_urg
  high_urg_level=31
  if model.find('no-urg')>=0:
    high_urg_level=1
  
  from utils import draw_errorbar
  fig=plt.figure(figsize=(24,12))
  axes=fig.subplots(2,2,sharex=True)
  

  data=np.zeros((len(urg_levels),len(coh_list),exp.repn,5))
  for j,i in enumerate(urg_levels):
    exp.A_urg=base_A_urg*i
  
    if not os.path.isdir(f'./urgency level {i}'):
        os.mkdir(f'./urgency level {i}')
    os.chdir(f'./urgency level {i}')

    if not eval_only:
        print(f'Running urgency level {i}')
        exp.run_experiment()
    
    print(f'Analyzing urgency level {i}')
    a=Analyzer(model=model,setting=setting,repn=repn,exp_maxt=exp_maxt,coh_list=coh_list)
    a.prepare()
    a.basic_analysis()
    a.draw_behavior()
    data[j,:,:,0]=a.correct
    data[j,:,:,1]=a.confidence
    data[j,:,:,2]=a.reaction_time
    data[j,:,:,3]=a.accunc
    data[j,:,:,4]=a.miss
      
    
    os.chdir(f'..')
    
  x=np.array(urg_levels)*base_A_urg
  for c in range(len(coh_list)):
    
    arr=[data[i,c,:,0] for i in range(data.shape[0])]
    
    draw_errorbar(arr,x=x,ax=axes[0][0],label=f'Coh level {c}',markersize=8)
    arr=[data[i,c,:,1][data[i,c,:,4]==0] for i in range(data.shape[0])]
    draw_errorbar(arr,x=x,ax=axes[0][1],label=f'Coh level {c}',markersize=8)
    arr=[data[i,c,:,2][data[i,c,:,4]==0] for i in range(data.shape[0])]
    draw_errorbar(arr,x=x,ax=axes[1][0],label=f'Coh level {c}',markersize=8)
    arr=[data[i,c,:,3][data[i,c,:,4]==0] for i in range(data.shape[0])]
    draw_errorbar(arr,x=x,ax=axes[1][1],label=f'Coh level {c}',markersize=8)
  axes[1][0].set_xlabel(r'$\alpha_{urg}$')
  axes[1][1].set_xlabel(r'$\alpha_{urg}$')
  from matplotlib.ticker import MaxNLocator
  axes[1][0].xaxis.set_major_locator(MaxNLocator(5)) 
  axes[1][1].xaxis.set_major_locator(MaxNLocator(5)) 
  
  axes[0][0].set_ylabel('Accuracy')
  axes[0][1].set_ylabel('Confidence')
  axes[1][0].set_ylabel('Reaction time (ms)')
  axes[1][1].set_ylabel('Uncertainty')

  axes[1][0].legend()


  fig.savefig('urg_para')

  