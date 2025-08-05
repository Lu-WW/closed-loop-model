

from analyzer import *
 
class PKAnalyzer(Analyzer):
    def __init__(self, load_data=True, load_detailed_data=True, **kwargs):
        super(PKAnalyzer, self).__init__(load_data=load_data,
                                       load_detailed_data=load_detailed_data, **kwargs)


    def cal_PK(self,decision_aligned=False,cohs=None):
        ## calculate Psychophysical Kernel (PK) across time point
        if cohs is None:
           cohs=np.arange(self.coh_level)
        if decision_aligned:
            self.get_decision_aligned_data()
            
            dat=self.decision_aligned_detailed_trial_data
        else:
            dat=self.detailed_trial_data
        evd=dat[:,:,self.id_sti[2]]-dat[:,:,self.id_sti[1]]

        ## remove coherence
        for c in range(evd.shape[0]):
            input_coh=self.coh_list[c]
            for r in range(evd.shape[1]):
              sti_on=(evd[c,r]!=0)&(~np.isnan(evd[c,r]))
              evd[c,r,sti_on]-=self.input0*input_coh*2
        evd=evd[cohs,:,:]
       
        evd=evd.reshape(-1,evd.shape[-1])
       

        y=self.choice==2
        y=y[cohs].flatten()
        from sklearn.metrics import roc_auc_score
        pret_idx = int(self.pret*self.sample_rate)
        l=[]
        for i in range(pret_idx,evd.shape[-1]):
          valid=(evd[:,i]!=0)&(~np.isnan(evd[:,i]))
          
          
          try:
            assert np.sum(valid)>100
            l.append(roc_auc_score(y[valid],evd[valid,i]))
          except:
            l.append(np.nan)

          
        if  decision_aligned:
          self.decision_aligned_pk=np.array(l)
          return self.decision_aligned_pk
        else:
            
          self.pk=np.array(l)

        return self.pk
       


from compare import *

def compare_pk(setting_list,name,decision_aligned=False,dashed_settings=[],label_list=None,cycler=None,figsize=(10, 7),legend=True,**kwargs):

   
    if not os.path.isdir(f'./compare/pk'):
        os.mkdir(f'./compare/pk')
    if label_list is None:
        label_list=setting_list
        
    print(f'\nDraw {name}')
    fig=plt.figure(figsize=figsize)
    
    
    ax=fig.subplots(1,1)
    from matplotlib.ticker import MaxNLocator


    ax.set_prop_cycle(cycler)
    ax.yaxis.set_major_locator(MaxNLocator(1)) 
    fig.tight_layout(h_pad=0.9)
    for i,setting in enumerate(setting_list):
        if not os.path.isdir(f'./{setting}'):
            print(f'No {setting} setting in {model} model')
            return
        os.chdir(f'./{setting}')

        print(f'  Draw {setting}')
       
        a=PKAnalyzer(model=model,setting=setting)
        pk=a.cal_PK(decision_aligned,**kwargs)
        x=np.array(range(pk.shape[-1]))/a.sample_rate*a.dt
        if decision_aligned:
           x=x-x[-1]
        if setting in dashed_settings:
          plt.plot(x,pk,label=label_list[i],ls='--',marker=',')
        else:
          plt.plot(x,pk,label=label_list[i],marker=',')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Stimulus impact (PK)')
        plt.axhline(0.5,color='k',ls='--',marker=',')
        os.chdir(f'..')
    if legend:
        plt.legend()
    fig.savefig(f'./compare/pk/{name}')
    return fig


   

if __name__=='__main__':
    import os
    model,setting,eval_only=init_with_args()
    os.chdir(f'..')

   
    compare_pk(['fixed_duration_experiment maxt_500_noise_1','fixed_duration_experiment maxt_1000_noise_1','fixed_duration_experiment maxt_3000_noise_1'],'pk_fixed_dur',label_list=['500 ms','1000 ms','3000 ms'])
    compare_pk(['fixed_duration_experiment maxt_500_noise_1','fixed_duration_experiment maxt_500_noise_2','fixed_duration_experiment maxt_500_noise_3'],'pk_fixed_noise',label_list=['Noise lv.1','Noise lv.2','Noise lv.3'])
