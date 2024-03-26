
import os
import numpy as np
import matplotlib.pyplot as plt
from analyzer import Analyzer

from utils import draw_errorbar, draw_bootstrap

def get_samp(setting_list,coh=None,only_choice=None,nsamp = 300):

  v1 = []
  v2 = []
  rt = []
  choice = []
  uncertainty = []
  confidence = []

  setting_samp=np.random.randint(0,len(setting_list),nsamp)
  setting_samp=[np.sum(setting_samp==i) for i in range(len(setting_list))]
  for i,setting in enumerate(setting_list):
    
    os.chdir(f'./{setting}')

    a = Analyzer(model=model, setting=setting)
    
    if setting.find('baseline') >= 0:
        baseline = float(setting.split('baseline_')[1])
    else:
        baseline = 0
    if setting.find('input') >= 0:
        a.input0 *= float(setting.split('input_')[1])

    if setting.find('noise') >= 0:
        noise = float(setting.split('noise_')[1])
    else:
        noise = 0
    s = 0
    while(s < setting_samp[i]):
      
      if coh:
        c=coh
      else:
        c = np.random.randint(1,a.coh_level)
        # c = np.random.randint(1,4)
        

      r = np.random.randint(a.repn)
      
      if a.miss[c,r]:
          continue
      if only_choice and a.choice[c,r] !=only_choice :
          continue
      s += 1
      chosen = (1+a.coh_list[c])*a.input0+baseline
      unchosen = (1-a.coh_list[c])*a.input0+baseline
      if a.choice[c,r] == 1:
          chosen, unchosen = unchosen, chosen
      # conf=a.confidence[c,r]
      # if conf<0:
      #   conf=0
      # chosen=np.log(chosen)
      # unchosen=np.log(unchosen)
      # chosen=np.exp(chosen)
      # unchosen=np.exp(unchosen)
      if noise!=0:
        v1.append(((chosen-unchosen)/(noise)))
        v2.append(((chosen+unchosen)/(noise)))
        rt.append(a.reaction_time[c,r])
        choice.append(a.correct[c,r])
        # confidence.append(conf)
        confidence.append(a.confidence[c,r])
        uncertainty.append(a.accunc[c,r])
      else:
        # for i in range(1,6):   
        #   v1.append(((chosen-unchosen)*i*100))
        #   v2.append(((chosen+unchosen)*i*100))
        #   rt.append(a.reaction_time[c,r])
        #   choice.append(a.correct[c,r])
        #   confidence.append(a.confidence[c,r])
        #   uncertainty.append(a.accunc[c,r])
        v1.append(((chosen-unchosen)*100))
        v2.append(((chosen+unchosen)*100))
        rt.append(a.reaction_time[c,r])
        choice.append(a.correct[c,r])
        # confidence.append(conf)
        confidence.append(a.confidence[c,r])
        uncertainty.append(a.accunc[c,r])
        
        # v1.append(((chosen-unchosen)*10))
        # v2.append(((chosen+unchosen)*10))
        # rt.append(a.reaction_time[c,r])
        # choice.append(a.correct[c,r])
        # confidence.append(a.confidence[c,r])
        # uncertainty.append(a.accunc[c,r])
          
    os.chdir(f'..')

  v1 = np.array(v1).flatten()
  v2 = np.array(v2).flatten()
  rt = np.array(rt).flatten()/1000
  
  choice = np.array(choice).flatten()
  uncertainty = np.array(uncertainty).flatten()
  confidence = np.array(confidence).flatten()
  
  m_uncertainty=[np.min(uncertainty),np.max(uncertainty)]
  m_confidence=[np.min(confidence),np.max(confidence)]
  # m_uncertainty=[0,50]
  # m_confidence=[-50,50]
  uncertainty=(uncertainty-m_uncertainty[0])/(m_uncertainty[1]-m_uncertainty[0])
  confidence=(confidence-m_confidence[0])/(m_confidence[1]-m_confidence[0])

  
  # uncertainty=(uncertainty-m_uncertainty[0])/(m_uncertainty[1]-m_uncertainty[0])
  # confidence=(confidence-m_confidence[0])/(m_confidence[1]-m_confidence[0])
  
  # uncertainty=np.log(uncertainty/(1-uncertainty))
  # confidence=np.log(confidence/(1-confidence))
  

  return v1,v2,rt,choice,uncertainty,confidence


def input_weight_regression(setting_list,coh=None,rep = 10000, eval_only=False,name='noisy',only_choice=None):

  for setting in setting_list:
      if not os.path.isdir(f'./{setting}'):
          print(f'No {setting} setting in {model} model')
          return
  acc_paras = []
  conf_paras = []
  unc_paras = []
  rt_paras=[]
  
  if not eval_only:
    
    for i in range(rep):
      v1,v2,rt,choice,uncertainty,confidence=get_samp(setting_list,coh=coh,only_choice=only_choice)
      import statsmodels.api as sm
      import pandas as pd

      x = np.stack((np.abs(v1), v2, rt), -1)
      x = pd.DataFrame(x)
      x = sm.add_constant(x)
      

      model_choice = sm.Logit(choice, x)
      results_choice = model_choice.fit(disp=0)
      acc_paras.append(results_choice.params)
      

      v1,v2,rt,choice,uncertainty,confidence=get_samp(setting_list)

      x = np.stack((v1, v2, rt), -1)
      x = pd.DataFrame(x)
      x = sm.add_constant(x)

      model_confidence = sm.GLM(confidence, x,family=sm.families.Binomial())
      
      results_confidence = model_confidence.fit()
      conf_paras.append(results_confidence.params)

      model_uncertainty = sm.GLM(uncertainty, x,family=sm.families.Binomial())
      
      results_uncertainty = model_uncertainty.fit()
      unc_paras.append(results_uncertainty.params)

      x = np.stack((v1, v2), -1)
      x = pd.DataFrame(x)
      x = sm.add_constant(x)
      
      model_rt = sm.GLM(rt, x)
      results_rt = model_rt.fit()
      rt_paras.append(results_rt.params)


      
    np.save(f'./GLM/{name}_acc_paras.npy', acc_paras)
    np.save(f'./GLM/{name}_conf_paras.npy', conf_paras)
    np.save(f'./GLM/{name}_unc_paras.npy', unc_paras)
    np.save(f'./GLM/{name}_rt_paras.npy', rt_paras)

  if (not os.path.isfile(f'./GLM/{name}_rt_paras.npy')) or (not os.path.isfile(f'./GLM/{name}_acc_paras.npy')) or (not os.path.isfile(f'./GLM/{name}_conf_paras.npy')) or (not os.path.isfile(f'./GLM/{name}_unc_paras.npy')):
    print(f'No regression data in {model} model')
    return
  acc_paras = np.load(f'./GLM/{name}_acc_paras.npy')[:rep]
  conf_paras = np.load(f'./GLM/{name}_conf_paras.npy')[:rep]
  unc_paras = np.load(f'./GLM/{name}_unc_paras.npy')[:rep]
  rt_paras = np.load(f'./GLM/{name}_rt_paras.npy')[:rep]
  

  para=[r'$\left|\Delta V\right|$',r'$\Sigma V$','RT']
  
  acc_paras=np.array(acc_paras)[:,1:1+len(para)]
  conf_paras=np.array(conf_paras)[:,1:1+len(para)]
  unc_paras=np.array(unc_paras)[:,1:1+len(para)]
  rt_paras=np.array(rt_paras)[:,1:1+len(para)]
  
  
  fig=plt.figure(figsize=(10,15))
  axes=fig.subplots(3,len(para))
  
  colors=['steelblue','limegreen','darkorange','steelblue','limegreen','darkorange']


  from utils import draw_violin
  from scipy.stats import ttest_1samp
  def add_sig(data,ax=plt,x=1,extend=1.3):
    ave=np.mean(data)
    if ave>0:
      
      p=np.mean(data<0)
    else:
      
      p=np.mean(data>0)
      

    w='bold'
    if p>0.05:
      sig='ns'
      w='normal'
    elif p>0.01:
      sig='*'
    elif p>0.001:
      sig='**'
    else:
      sig='***'
    
    m=extend*np.max(np.abs(ax.get_ylim()))
    ax.set_ylim(-m,m) 
    if ave>0:
      ax.text(x,0.85*m,sig,ha='center',weight=w)
    else:
      ax.text(x,-0.95*m,sig,ha='center',weight=w)
       
    
    return p
       
  for i in range(acc_paras.shape[1]):
    draw_violin(acc_paras[:,i],axes[0][i],xlabel=para[i],color=colors[i])
    p=add_sig(acc_paras[:,i],axes[0][i])
    
    
    
  para=[r'$\Delta V$',r'$\Sigma V$','RT']
  

  for i in range(conf_paras.shape[1]):
    
    draw_violin(conf_paras[:,i],axes[1][i],xlabel=para[i],color=colors[i])
    p=add_sig(conf_paras[:,i],axes[1][i])

  for i in range(unc_paras.shape[1]):
    draw_violin(unc_paras[:,i],axes[2][i],xlabel=para[i],color=colors[i])
    p=add_sig(unc_paras[:,i],axes[2][i])


  axes[0][0].set_ylabel('Accuracy coefficient',labelpad=10.0)
  axes[1][0].set_ylabel('Confidence coefficient',labelpad=10.0)
  axes[2][0].set_ylabel('Uncertainty coefficient',labelpad=10.0)
  

  for i in range(3):
    for j in range(len(para)):
      axes[i][j].ticklabel_format(style='sci', scilimits=(-1,2), axis='both')    
      axes[i][j].set_xticks([])
      m=np.max(np.abs(axes[i][j].get_ylim()))
      axes[i][j].set_ylim(-m,m)
      
      

  fig.tight_layout(w_pad=3)
  fig.savefig(f'./GLM/{name}_para_hist')


  fig=plt.figure()
  para=[r'$\Delta V$',r'$\Sigma V$']
  axes=fig.subplots(1,len(para))
  
  for i in range(rt_paras.shape[1]):
    draw_violin(rt_paras[:,i],axes[i],xlabel=para[i],color=colors[i])
    p=add_sig(rt_paras[:,i],axes[i])
  fig.savefig(f'./GLM/{name}_rt_para_hist')

if __name__ == '__main__':
  from utils import init_with_args
  model, setting, eval_only = init_with_args()
  os.chdir(f'..')
  
  
  setting_list=['noise_0.5','noise_1','noise_1.5']
  
  if model!='no-urg-mi' and model!='urg-on-dec-mi':
     setting_list=['noise_1','noise_2','noise_3',]
     
  input_weight_regression(setting_list,rep=100,eval_only=eval_only,name='try100')
  