
import os
import numpy as np
import matplotlib.pyplot as plt
from analyzer import Analyzer

from utils import draw_errorbar,draw_bootstrap

def draw_compare(setting_list,name,dashed_settings=[],label_list=None,cycler=None,figsize=(24, 12),coh_level=None,nsamp=None,legend_subplot=(0,0),legend=True,**kwargs):
    ## compare basic behavioral results (choice, reaction time, confidence, uncertainty)
    if label_list is None:
        label_list=setting_list
    print(f'\nDraw {name}')
    fig=plt.figure(figsize=figsize)
    
    axes=fig.subplots(2,2)
    for ax in axes.flatten():
       ax.set_prop_cycle(cycler)
    for i,setting in enumerate(setting_list):

        if not os.path.isdir(f'./{setting}'):
            print(f'No {setting} setting in {model} model')
            return
        os.chdir(f'./{setting}')

        print(f'  Draw {setting}')
       
        a=Analyzer(model=model,setting=setting,load_detailed_data=False)
        if nsamp is not None:
            a.trial_data=a.trial_data[:,:nsamp]
        a.prepare()
        if coh_level is None:
            coh_level=a.coh_level
        
        linestyle='solid'
        if setting in dashed_settings:
            linestyle='dashed'
        kwargs['linestyle']=linestyle

        arr=a.correct[:coh_level]
        draw_errorbar(arr,ax=axes[0,0],label=f'{label_list[i]}',**kwargs)
      
        ####
        arr=[a.confidence[coh][~a.miss[coh]] for coh in range(coh_level)]
        draw_errorbar(arr,ax=axes[0,1],label=f'{label_list[i]}',**kwargs)
        

        arr=[(a.reaction_time[coh][~a.miss[coh]]) for coh in range(coh_level)]

        draw_errorbar(arr,ax=axes[1,0],label=f'{label_list[i]}',**kwargs)
        

        arr=[a.accunc[coh][~a.miss[coh]] for coh in range(coh_level)]
        
        draw_errorbar(arr,ax=axes[1,1],label=f'{label_list[i]}',**kwargs)


        os.chdir('..')

    # axes[0,0].set_xlabel('Coherence level')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_xticks(range(coh_level))

    # axes[0,1].set_xlabel('Coherence level')
    axes[0,1].set_ylabel('Confidence')
    axes[0,1].set_xticks(range(coh_level))
    
    axes[1,0].set_xlabel('Coherence level')
    axes[1,0].set_ylabel('Reaction Time (ms)')
    axes[1,0].set_xticks(range(coh_level))

    axes[1,1].set_xlabel('Coherence level')
    axes[1,1].set_ylabel('Uncertainty')
    axes[1,1].set_xticks(range(coh_level))

    if legend:
        axes[legend_subplot].legend(markerscale=0)
        

    fig.savefig(f'./compare/{name}')
    return fig

def draw_compare_ave(setting_list,name,mode='dif',dashed_settings=[],correct_only=False,label_list=None,coh=-1,cycler=None,figsize=(10, 7),window_length=300,legend_subplot=(3,0),legend=True,**kwargs):
    ## compare average activity
   
    if label_list is None:
        label_list=setting_list
        
    print(f'\nDraw {name}')
    fig=plt.figure(figsize=figsize)
    
    axes=fig.subplots(4,2)
    from matplotlib.ticker import MaxNLocator

    for ax in axes.flatten():
       ax.set_prop_cycle(cycler)
       ax.yaxis.set_major_locator(MaxNLocator(1)) 
    fig.tight_layout(h_pad=0.9)
    for i,setting in enumerate(setting_list):
        if not os.path.isdir(f'./{setting}'):
            print(f'No {setting} setting in {model} model')
            return
        os.chdir(f'./{setting}')

        print(f'  Draw {setting}')
       
        a=Analyzer(model=model,setting=setting)
        a.prepare()
        
        linestyle='solid'
        if setting in dashed_settings:
            linestyle='dashed'
        kwargs['linestyle']=linestyle



        def get_data(a,decision_aligned):
            npoint=int(np.ceil((window_length+200)/a.dt/a.maxt*a.nsamp))
            if decision_aligned:
                a.get_decision_aligned_data()
                data=a.decision_aligned_detailed_trial_data[...,-npoint:]
                timex=np.linspace(-(npoint-1)/a.nsamp*a.maxt*a.dt,0,npoint)+a.decision_aligned_delay
            else:
                timex=np.array(range(a.nsamp))/a.nsamp*a.maxt*a.dt-a.pret*a.dt
                pre_delay=int(300/a.dt/a.maxt*a.nsamp)
                data=a.detailed_trial_data
                data=data[...,:pre_delay+npoint]
                timex=timex[...,:pre_delay+npoint]
            return data,timex
                
        def draw(a,axes,col,decision_aligned,coh):
            data,timex = get_data(a,decision_aligned)
            idx=~a.miss ##
            if correct_only:
                idx=a.choice == 2
            def get_data_by_mode(mode,a,id1,id2):
                import copy
                if mode=='dif':
                    dat=data[:,:,id2]-data[:,:,id1] 
                    dat[a.choice == 1]*=-1
                elif mode=='chosen':
                    dat=copy.deepcopy(data[:,:,id2])
                    idx=a.choice == 1
                    dat[idx]=data[idx][:,id1] 
                elif mode=='unchosen':
                    dat=copy.deepcopy(data[:,:,id1])
                    idx=a.choice == 1
                    dat[idx]=data[idx][:,id2] 
                elif mode=='correct':
                    dat=copy.deepcopy(data[:,:,id2])
                elif mode=='incorrect':
                    dat=copy.deepcopy(data[:,:,id1])
                elif mode=='all':
                    dat=np.concatenate((data[:,:,id1],data[:,:,id2]),1) 
                else:
                    print(f'Unknown mode {mode}\nUse dif instead\n')
                    dat=data[:,:,id2]-data[:,:,id1] 
                    dat[a.choice == 1]*=-1
                return dat
                
            if type(coh)!=int or coh==-1:
                if type(coh)==int:
                    coh=range(a.coh_level)
                dat=get_data_by_mode(mode,a,a.id_decision[1],a.id_decision[2])
                for c in coh:
                    arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                    draw_errorbar(arr,x=timex,ax=axes[0,col],label=f'Coh {(a.coh_list[c]):.1%}',marker=',',**kwargs)
                    
                dat=get_data_by_mode(mode,a,a.id_motor[1],a.id_motor[2])
                for c in coh:
                    arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                    draw_errorbar(arr,x=timex,ax=axes[1,col],label=f'Coh {(a.coh_list[c]):.1%}',marker=',',**kwargs)

                dat=data[:,:,a.id_unc[0]]
                for c in coh:
                    arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                    draw_errorbar(arr,x=timex,ax=axes[2,col],label=f'Coh {(a.coh_list[c]):.1%}',marker=',',**kwargs)

                dat=data[:,:,a.id_accunc[0]]
                for c in coh:
                    arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                    draw_errorbar(arr,x=timex,ax=axes[3,col],label=f'Coh {(a.coh_list[c]):.1%}',marker=',',**kwargs)
            
            else:
                c=coh
                
                dat=get_data_by_mode(mode,a,a.id_decision[1],a.id_decision[2])
                arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=timex,ax=axes[0,col],label=f'{label_list[i]}',marker=',',**kwargs)
                    
                dat=get_data_by_mode(mode,a,a.id_motor[1],a.id_motor[2])
                arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=timex,ax=axes[1,col],label=f'{label_list[i]}',marker=',',**kwargs)
                

                dat=data[:,:,a.id_unc[0]]
                arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=timex,ax=axes[2,col],label=f'{label_list[i]}',marker=',',**kwargs)

                dat=data[:,:,a.id_accunc[0]]
                arr=[dat[c,:,t][idx[c]] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=timex,ax=axes[3,col],label=f'{label_list[i]}',marker=',',**kwargs)
        draw(a,axes,0,False,coh)
        draw(a,axes,1,True,coh)

                

        
        
        os.chdir('..')
    for ax in axes[:,0]:
        ax.set_xlim(-50,window_length-50)
    for ax in axes[:,1]:
        ax.set_xlim(50-window_length,50)
        
    if mode=='dif':
        axes[0,0].set_ylabel(r'$\Delta$ Dec')
        axes[1,0].set_ylabel(r'$\Delta$ Mot')
    elif mode=='chosen':
        axes[0,0].set_ylabel('Chosen\nDec')
        axes[1,0].set_ylabel('Chosen\nMot')
    elif mode=='unchosen':
        axes[0,0].set_ylabel('Unchosen\nDec')
        axes[1,0].set_ylabel('Unchosen\nMot')
    elif mode=='correct':
        axes[0,0].set_ylabel('Correct\nDec')
        axes[1,0].set_ylabel('Correct\nMot')
    elif mode=='incorrect':
        axes[0,0].set_ylabel('Incorrect\nDec')
        axes[1,0].set_ylabel('Incorrect\nMot')
    elif mode=='all':
        axes[0,0].set_ylabel(r'Dec')
        axes[1,0].set_ylabel(r'Mot')



    axes[2,0].set_ylabel('Ins-Unc')
    

    axes[3,0].set_xlabel('Stimulus onset aligned Time (ms)')
    axes[3,0].set_ylabel('Acc-Unc')

    axes[3,1].set_xlabel('Decision aligned time (ms)')
    for ax in axes.flatten():
        ax.axvline(0,linestyle='dashed',color='black',marker=',')
        
        
    if legend:
        axes[legend_subplot].legend()
    fig.align_ylabels(axes[:,0])
    fig.align_ylabels(axes[:,1])
    fig.savefig(f'./compare/{name}')
    return fig

def cross_analyze_metacog(setting_list,name,draw_mixed=False,coh=None,dashed_settings=[],nbin=20,label_list=None,cycler=None,figsize=(21, 14),legend_subplot=(0,0),legend=True,discard_extremes=False,**kwargs):
    ## compare relations between behavioral variables
    
    if label_list is None:
        label_list=setting_list
    if coh is not None:
        name+=f'_coh_level_{coh}'
    print(f'\nAnalyzing meta-cognition with {name}')
    from utils import vs_quantile
    fig=plt.figure(figsize=figsize)
    axes=fig.subplots(2,3)
    for ax in axes.flatten():
       ax.set_prop_cycle(cycler)
       
    mixed_confidence=np.zeros(0)
    mixed_correct=np.zeros(0)
    mixed_rt=np.zeros(0)
    mixed_accunc=np.zeros(0)
    for i,setting in enumerate(setting_list):
        if not os.path.isdir(f'./{setting}'):
            print(f'No {setting} setting in {model} model')
            return
        os.chdir(f'./{setting}')


        linestyle='solid'
        if setting in dashed_settings:
            linestyle='dashed'
        kwargs['linestyle']=linestyle


        a=Analyzer(model=model,setting=setting,load_detailed_data=False)
        a.prepare()

        idx=~a.miss
        if coh:
            idx[:coh]=0
            idx[coh+1:]=0
        confidence=a.confidence[idx]
        correct=a.correct[idx]
        rt=a.reaction_time[idx]
        accunc=a.accunc[idx]
        if draw_mixed:
            mixed_confidence=np.concatenate((mixed_confidence,confidence))
            mixed_correct=np.concatenate((mixed_correct,correct))
            mixed_rt=np.concatenate((mixed_rt,rt))
            mixed_accunc=np.concatenate((mixed_accunc,accunc))
            
        def draw(confidence,correct,rt,accunc,label=None):
            x,arr = vs_quantile(confidence,correct,nbin=nbin,discard_extremes=discard_extremes)
            draw_errorbar(arr,x=x,ax=axes[0][0],label=label,markersize=9,**kwargs)
            axes[0][0].set_xlabel('Confidence')
            axes[0][0].set_ylabel('Accuracy')
            
            
            x,arr = vs_quantile(accunc,correct,nbin=nbin,discard_extremes=discard_extremes)
            draw_errorbar(arr,x=x,ax=axes[0][1],markersize=9,**kwargs)
            axes[0][1].set_xlabel('Uncertainty')
            axes[0][1].set_ylabel('Accuracy')
            
            
           


            x,arr = vs_quantile(confidence,accunc,nbin=nbin,discard_extremes=discard_extremes)
            draw_errorbar(arr,x=x,ax=axes[0][2],markersize=9,**kwargs)
            axes[0][2].set_xlabel('Confidence')
            axes[0][2].set_ylabel('Uncertainty')



            x,arr = vs_quantile(rt,correct,nbin=nbin,discard_extremes=discard_extremes)
            draw_errorbar(arr,x=x,ax=axes[1][0],markersize=9,**kwargs)
            axes[1][0].set_xlabel('Reaction time (ms)')
            axes[1][0].set_ylabel('Accuracy')
            
            x,arr = vs_quantile(rt,confidence,nbin=nbin,discard_extremes=discard_extremes)
            draw_errorbar(arr,x=x,ax=axes[1][1],markersize=9,**kwargs)
            axes[1][1].set_xlabel('Reaction time (ms)')
            axes[1][1].set_ylabel('Confidence')
            
            
            x,arr = vs_quantile(rt,accunc,nbin=nbin,discard_extremes=discard_extremes)
            draw_errorbar(arr,x=x,ax=axes[1][2],markersize=9,**kwargs)
            axes[1][2].set_xlabel('Reaction time (ms)')
            axes[1][2].set_ylabel('Uncertainty')



        
            # x,arr = vs_quantile(confidence,accunc,nbin=nbin,discard_extremes=discard_extremes)
            # draw_errorbar(arr,x=x,ax=extra_axes,markersize=9,**kwargs)
            # extra_axes.set_xlabel('Confidence')
            # extra_axes.set_ylabel('Uncertainty')


            # x,arr = vs_quantile(rt,correct,nbin=nbin,discard_extremes=discard_extremes)
            # draw_errorbar(arr,x=x,ax=extra_axes2,markersize=9,**kwargs)
            # extra_axes2.set_xlabel('Reaction time')
            # extra_axes2.set_ylabel('Accuracy')


        draw(confidence,correct,rt,accunc,label=label_list[i])
        os.chdir('..')
        
    if draw_mixed:
        draw(mixed_confidence,mixed_correct,mixed_rt,mixed_accunc,label='Mixed')
        
    fig.tight_layout(pad=2)
    if legend:
        axes[legend_subplot].legend()
    if not os.path.isdir(f'./compare/{name}'):
        os.mkdir(f'./compare/{name}')
    os.chdir(f'./compare/{name}')
    
    fig.savefig(f'meta_cog')
    
    
    os.chdir(f'..')
    os.chdir(f'..')
    return fig



def draw_compare_two_side(setting_list,name,dashed_settings=[],label_list=None,short_label_list=None,cycler=None,figsize=(16, 12),coh_level=None,nsamp=None,legend_subplot=(0,1),legend=True,**kwargs):
    ## compare behavioral results but assuming paired settings 2x & 2x+1 corresponds to stimulus supporting different choices
    ## e.g., setting_list = ['normal','normal','baseline motor','baseline motor_weak',]
    if label_list is None:
        label_list=setting_list
    if short_label_list is None:
        short_label_list=label_list
    print(f'\nDraw {name}')
    fig=plt.figure(figsize=figsize)
    
    axes=fig.subplots(2,2)
    for ax in axes.flatten():
       ax.set_prop_cycle(cycler)

    curves=[]
    for i,setting in enumerate(setting_list):

        if not os.path.isdir(f'./{setting}'):
            print(f'No {setting} setting in {model} model')
            return
        os.chdir(f'./{setting}')

        print(f'  Draw {setting}')
       
        a=Analyzer(model=model,setting=setting,load_detailed_data=False)
        if nsamp is not None:
            a.trial_data=a.trial_data[:,:nsamp]
        a.prepare()
        if coh_level is None:
            coh_level=a.coh_level
        
        linestyle='solid'
        if setting in dashed_settings:
            linestyle='dashed'
        kwargs['linestyle']=linestyle
        x=np.array(a.coh_list)[:coh_level]
        if i%2==0:
            x=-x
            a.correct=1-a.correct
        arr=a.correct[:coh_level]
        if i%2==0:  
            cohs=np.tile(x.reshape((-1,1)),a.correct.shape[1])
            curve=[(cohs.flatten(),arr.flatten())]
            other_side_data=a.trial_data[::-1]
        else:
            
            cohs=np.tile(x.reshape((-1,1)),a.correct.shape[1]).flatten()
            y=arr.flatten()
            
            cohs=np.concatenate((cohs,curve[0][0]))
            y=np.concatenate((y,curve[0][1]))
            def glm(x,y):
              import statsmodels.api as sm
              import pandas as pd
              x=pd.DataFrame(x,columns=['x'])
              x = sm.add_constant(x)
              glm=sm.GLM(y,x,family=sm.families.Binomial())
              res=glm.fit()
              
              
              return res.params[1],res.params[0]
            # print(label_list[int(i/2)])
            k,b=glm(cohs,y)
            curves.append((k,b))
            # print(k,b)

        if i%2!=0:
            a.trial_data=np.concatenate((other_side_data,a.trial_data[1:]),0)  
            a.prepare()
            coh_level=a.trial_data.shape[0]
            # x=np.linspace(-np.floor(coh_level/2),np.floor(coh_level/2),coh_level).astype(int)
            x=np.unique(cohs)
            

            a.correct[:other_side_data.shape[0]]=1-a.correct[:other_side_data.shape[0]]
            arr=[a.correct[coh] for coh in range(coh_level)]
            draw_errorbar(arr,x=x,ax=axes[0,0],label=f'{label_list[int(i/2)]}',**kwargs)
            
            ####
            arr=[a.confidence[coh][~a.miss[coh]] for coh in range(coh_level)]
            draw_errorbar(arr,x=x,ax=axes[0,1],label=f'{label_list[int(i/2)]}',**kwargs)
            

            arr=[(a.reaction_time[coh][~a.miss[coh]])*a.dt for coh in range(coh_level)]

            draw_errorbar(arr,x=x,ax=axes[1,0],label=f'{label_list[int(i/2)]}',**kwargs)
            

            arr=[a.accunc[coh][~a.miss[coh]] for coh in range(coh_level)]
            
            draw_errorbar(arr,x=x,ax=axes[1,1],label=f'{label_list[int(i/2)]}',**kwargs)


        os.chdir('..')
    # x=np.array(a.coh_list)[:coh_level]
    # x=np.concatenate((-x[1:][::-1],x))
    x=[-0.5,0,0.5]
    # axes[0,0].set_xlabel('Coherence level')
    axes[0,0].set_ylabel('Proportion of right choice')
    axes[0,0].set_xticks(x)

    # axes[0,1].set_xlabel('Coherence level')
    axes[0,1].set_ylabel('Confidence')
    axes[0,1].set_xticks(x)
    
    axes[1,0].set_xlabel('Coherence')
    axes[1,0].set_ylabel('Reaction Time (ms)')
    axes[1,0].set_xticks(x)

    axes[1,1].set_xlabel('Coherence')
    axes[1,1].set_ylabel('Uncertainty')
    axes[1,1].set_xticks(x)

    if legend:
        axes[legend_subplot].legend(loc=(0.3,0.8),markerscale=0)
        
    fig.subplots_adjust(wspace=0.3)
    
    slope_ax=fig.add_axes([0.35,0.6,0.12,0.1],)
    slope_ax.set_prop_cycle(cycler)
    slope_ax.plot(range(len(short_label_list)),[k for k,b in curves],color='black',marker=',')
    for i in range(len(short_label_list)):
        slope_ax.scatter(i,curves[i][0],zorder=3)
    slope_ax.margins(0.15)
    slope_ax.set_xticks(range(len(short_label_list)),short_label_list)
    slope_ax.set_ylabel('Slope')


    fig.savefig(f'./compare/{name}')
    return fig




def draw_comparison(model_list,name,model_names=None,dashed_models=[],settings=['normal','noise_1','noise_2','noise_3'],label_list=None,cycler=None,figsize=(24, 20),coh_level=None,nsamp=None,legend_subplot=(0,0),legend=True,**kwargs):
    ## compare behavioral results across models in different settings
    if label_list is None:
        label_list=settings
    if model_names is None:
        model_names=model_list
        
    ori_path=os.getcwd()
    img_path=os.path.join(ori_path,'pics')
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    print(f'\nDraw {name}')
      
    fig=plt.figure(figsize=figsize)
    

    n=len(settings)
    axes=fig.subplots(4,n,sharex=True,sharey='row')
    for ax in axes.flatten():
       ax.set_prop_cycle(cycler)

    for i,model in enumerate(model_list):

        print(f'  Draw {model}')
        try:
          os.chdir(os.path.join(ori_path,model))
        except:
          print(f'No model named {model}')
          continue
        for j,setting in enumerate(settings):
          if not os.path.isdir(f'./{setting}'):
              print(f'No {setting} setting in {model} model')
              return
          os.chdir(f'./{setting}')


          a=Analyzer(model=model,setting=setting,load_detailed_data=False)
          if nsamp is not None:
              a.trial_data=a.trial_data[:,:nsamp]
          a.prepare()
          if coh_level is None:
              coh_level=a.coh_level
          
          linestyle='solid'
          if model in dashed_models:
            linestyle='dashed'
          kwargs['linestyle']=linestyle


          arr=a.correct[:coh_level]
          draw_errorbar(arr,ax=axes[0,j],label=f'{model_names[i]}',**kwargs)
          

          arr=[(a.reaction_time[coh][~a.miss[coh]])*a.dt for coh in range(coh_level)]

          draw_errorbar(arr,ax=axes[1,j],label=f'{model_names[i]}',**kwargs)
          
          def zscore(x):
            l=[]
            for a in x:
              l.extend(a.flatten().tolist()) 
            m=np.mean(l)
            s=np.std(l)
            l=[]  
            for a in x:
              l.append((a-m)/s)       
            return l
          ####
          arr=zscore([a.confidence[coh][~a.miss[coh]] for coh in range(coh_level)])
          draw_errorbar(arr,ax=axes[2,j],label=f'{model_names[i]}',**kwargs)
          

          arr=zscore([a.accunc[coh][~a.miss[coh]] for coh in range(coh_level)])
          draw_errorbar(arr,ax=axes[3,j],label=f'{model_names[i]}',**kwargs)


          os.chdir('..')
          
    

    axes[0,0].set_ylabel('Accuracy')
    axes[1,0].set_ylabel('Reaction Time (ms)')
    axes[2,0].set_ylabel('Normalized confidence')
    axes[3,0].set_ylabel('Normalized uncertainty')
    

    for j in range(n):
      axes[0,j].set_title(label_list[j])
      axes[3,j].set_xlabel('Coherence level')
      axes[3,j].set_xticks(range(coh_level))

    if legend:
        axes[legend_subplot].legend(markerscale=0,loc=[0.2,1.2])
        
    fig.align_labels()
    fig.savefig(os.path.join(img_path,name))
    
    os.chdir(ori_path)
    return fig





if __name__=='__main__':
    
    ori_path=os.getcwd()
    from utils import init_with_args
    model,setting,eval_only=init_with_args()
    os.chdir(f'..')

    
    ## unilateral inactivation
    from cycler import cycler
    c=cycler('color', ['cornflowerblue','darkorange',])
    draw_compare_two_side(['normal','normal','baseline motor','baseline motor_weak',],'compare_motor_baseline_twoside',label_list=['No inactivation','Motor inactivation',],short_label_list=['Intact','Inactivated',],cycler=c)    
    c=cycler('color', plt.cm.YlGnBu(np.linspace(0.1,0.9,7)))
    

    ## urgency signal
    c=cycler('color', plt.cm.Greens(np.linspace(0.3,0.6,2)))
    
    draw_compare(['normal','long'],'compare_sat_example',figsize=(20,10),cycler=c,label_list=[r'Speed($\alpha_{urg}=0.002$)',r'Accuracy($\alpha_{urg}=0$)'])
    draw_compare_ave(['normal','long'],'compare_sat_example_ave',coh=3,figsize=(20,10),cycler=c,legend=False,mode='chosen')
    cross_analyze_metacog(['normal','long'],'sat_example',cycler=c,label_list=[r'Speed($\alpha_{urg}=0.002$)',r'Accuracy($\alpha_{urg}=0$)'])
    

    ## noise level
    c=cycler('color', plt.cm.Blues(np.linspace(0.2,1,4)))
    figsize=(14,7)
    legend_subplot=(0,1)
    legend=True
    draw_compare(['normal','noise_1','noise_2','noise_3'],'compare_noise',figsize=figsize,cycler=c,label_list=['No noise','Noise lv.1','Noise lv.2','Noise lv.3'],legend_subplot=legend_subplot,legend=legend)
    draw_compare_ave(['normal','noise_1','noise_2','noise_3'],'compare_noise_ave',coh=3,cycler=c,label_list=['No noise','Noise lv.1','Noise lv.2','Noise lv.3'])
    cross_analyze_metacog(['normal','noise_1','noise_2','noise_3'],'different_noise',cycler=c,label_list=['No noise','Noise lv.1','Noise lv.2','Noise lv.3'])
    




    ## reverse task
    figsize=(24,16)
    c=cycler('color', np.concatenate((plt.cm.Blues(np.linspace(0.2,1,4)[1:]),plt.cm.Reds(np.linspace(0.2,1,4)[1:]))))

    from matplotlib.legend_handler import HandlerTuple
    fig=draw_compare(['noise_1','noise_2','noise_3','reverse noise_1','reverse noise_2','reverse noise_3'],'compare_reverse_noisy',legend=False,figsize=figsize,cycler=c)
    axes=fig.get_axes()
    h,_=axes[0].get_legend_handles_labels()
    axes[0].legend(handles=[(h[0],h[3]),(h[1],h[4]),(h[2],h[5])],labels=['Noise lv.1','Noise lv.2','Noise lv.3'],handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.savefig('./compare/compare_reverse_noisy')
 
    fig=cross_analyze_metacog(['noise_1','noise_2','noise_3','reverse noise_1','reverse noise_2','reverse noise_3'],'reverse_noisy',legend=False,cycler=c)
    axes=fig.get_axes()
    h,_=axes[0].get_legend_handles_labels()
    axes[0].legend(handles=[(h[0],h[3]),(h[1],h[4]),(h[2],h[5])],labels=['Noise lv.1','Noise lv.2','Noise lv.3'],handler_map={tuple: HandlerTuple(ndivide=None)})
    
    fig.savefig(f'./compare/reverse_noisy/meta_cog')


    
    figsize=(14,7)
    legend_subplot=(1,0)
    # draw_compare(['normal','baseline decision_both'],'compare_decision_baseline_both',dashed_settings=['baseline decision_both'],label_list=['No inactivation','Decision inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    # draw_compare(['normal','baseline motor_both'],'compare_motor_baseline_both',dashed_settings=['baseline motor_both'],label_list=['No inactivation','Motor inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    # draw_compare(['normal','baseline decision_weak'],'compare_decision_baseline_weak',dashed_settings=['baseline decision_weak'],label_list=['No inactivation','Decision inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    # draw_compare(['normal','baseline motor_weak'],'compare_motor_baseline_weak',dashed_settings=['baseline motor_weak'],label_list=['No inactivation','Motor inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    
    ## module inactivation
    draw_compare(['normal','baseline decision'],'compare_decision_baseline',dashed_settings=['baseline decision'],label_list=['No inactivation','Decision inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    draw_compare(['normal','baseline unc'],'compare_unc_baseline',dashed_settings=['baseline unc'],label_list=['No inactivation','Ins-unc inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    draw_compare(['normal','baseline au'],'compare_accunc_baseline',dashed_settings=['baseline au'],label_list=['No inactivation','Acc-unc inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    draw_compare(['normal','baseline motor'],'compare_motor_baseline',dashed_settings=['baseline motor'],label_list=['No inactivation','Motor inactivation'],legend_subplot=legend_subplot,shaded=False,figsize=figsize)
    draw_compare_ave(['normal','baseline decision'],'compare_decision_baseline_ave',coh=3,dashed_settings=['baseline decision'],label_list=['No inactivation','Decision inactivation'],shaded=False,legend=False)
    draw_compare_ave(['normal','baseline unc'],'compare_unc_baseline_ave',coh=3,dashed_settings=['baseline unc'],label_list=['No inactivation','Ins-unc inactivation'],shaded=False,legend=False)
    draw_compare_ave(['normal','baseline au'],'compare_accunc_baseline_ave',coh=3,dashed_settings=['baseline au'],label_list=['No inactivation','Acc-unc inactivation'],shaded=False,legend=False)
    draw_compare_ave(['normal','baseline motor'],'compare_motor_baseline_ave',coh=3,dashed_settings=['baseline motor'],label_list=['No inactivation','Motor inactivation'],shaded=False,legend=False)

    
    
    ## compare models

    # os.chdir(ori_path)
    # models=['closed-loop','urg-on-dec-mi','no-urg-mi']
    # names=['Full model','No-motor model','No-motor-urgency model']
    
    # c=cycler('color', plt.cm.YlGnBu(np.linspace(1,0.3,len(models))))
    # draw_comparison(models,f'model_comparison',model_names=names,dashed_models=models[1:],label_list=['No noise','Noise lv.1','Noise lv.2','Noise lv.3'],cycler=c,alpha=0.9,shaded=False,markersize=12,mew=0)
      