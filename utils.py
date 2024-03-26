

import numpy as np
import matplotlib.pyplot as plt
def draw_errorbar(arr,ax=plt,x=[],shaded=True,**kwargs):
    if np.size(x)==0:
        x=range(len(arr))
        
    y=[np.mean(a) for a in arr]
    y_bar=[np.std(a) for a in arr]/np.sqrt(np.array([len(a) for a in arr]))
    
    l,=ax.plot(x,y,**kwargs)
    if shaded:
        ax.fill_between(x,y-y_bar,y+y_bar,alpha=0.2,color=l.get_color())
    
    return y,y_bar

def draw_bootstrap(arr1,arr2,ax=plt,x=[],rep=1000,**kwargs):
    if np.size(x)==0:
        x=range(len(arr1))
        
    y=np.array([np.mean(arr1[i])-np.mean(arr2[i]) for i in range(len(arr1))])
    bootstrap_list=[]
    for i in range(len(arr1)):
        tmp_list=[]
        for j in range(rep):
            delt=np.mean(arr1[i][np.random.randint(0,arr1[i].shape[0],arr1[i].shape[0])])-np.mean(arr2[i][np.random.randint(0,arr2[i].shape[0],arr2[i].shape[0])])
            tmp_list.append(delt)
        bootstrap_list.append(tmp_list)
            
    y_bar=np.array([np.std(a) for a in bootstrap_list])
    # for i,a in enumerate(bootstrap_list):
        
    #     from scipy.stats import ttest_1samp
    #     s,p=ttest_1samp(a,0)
    #     print(i,np.mean(a),np.std(a),p)
    
    l,=ax.plot(x,y,**kwargs)

    ax.fill_between(x,y-y_bar,y+y_bar,alpha=0.2,color=l.get_color())
    
    return y,y_bar

def draw_violin(data,ax,xlabel='',ylabel='',color='grey'):
    violin=ax.violinplot(data,showextrema=False)
    
    for patch in violin['bodies']:
      patch.set_facecolor(color)
      patch.set_linewidth(0)
      patch.set_alpha(0.5)
      verts=patch.get_paths()[0].vertices
      poly=plt.Polygon(verts,color='darkgrey',alpha=1,lw=2,fill=False)
      ax.add_patch(poly)
      
      
    v5,v25, v50, v75, v95 = np.percentile(data, [5, 25, 50, 75, 95])
    ax.scatter(1,v50,color='white',s=50,zorder=4)
    ax.vlines(1,v25,v75, lw=12, zorder=3,color=color)
    ax.vlines(1,v5,v95, lw=4, zorder=2,color=color)

    ax.axhline(0,color='black',linestyle='dashed',marker=',',linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def show_run(exp,r,rt=0,nplt=300):
    
    endt=int(rt-exp.pret)
    timex=np.array(range(r.shape[-1]-exp.pret))*exp.dt
   
   
    from utils import down_sample
    
    timex_down_sampled=down_sample(timex,nplt)
    fig=plt.figure(figsize=(14, 12))
    
    plt.subplot(221)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_sti[1], exp.pret:],nplt), label='Left',marker=',',color=exp.stim_color)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_sti[2], exp.pret:],nplt), label='Right',marker=',',color=exp.stim_dark_color)
    # plt.ylim(0.01,0.02)
    if rt>0:
        plt.axvline(endt*exp.dt,color='black',linestyle='dashed',marker=',')
    plt.legend()
    plt.title('Stimulus')
    plt.ylabel('Current (nA)')
    
    plt.subplot(222)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_decision[1], exp.pret:],nplt), label='Left',marker=',',color=exp.decision_color)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_decision[2], exp.pret:],nplt), label='Right',marker=',',color=exp.decision_dark_color)
    
    if rt>0:
        plt.axvline(endt*exp.dt,color='black',linestyle='dashed',label='Trial end',marker=',')
    plt.legend()
    
    plt.title('Decision')
    plt.ylabel('Firing rate (Hz)')
    plt.subplot(223)
    plt.plot(0,0,marker=',')
    plt.plot(0,0,marker=',')
    plt.plot(timex_down_sampled,down_sample(r[exp.id_unc[0], exp.pret:],nplt), label='Instantaneous',marker=',',color=exp.insunc_color)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_accunc[0], exp.pret:],nplt), label='Accumulated',marker=',',color=exp.accunc_color)
    plt.axvline(endt*exp.dt,color='black',linestyle='dashed',marker=',')
    plt.legend()
    plt.title('Uncertainty')
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')

    plt.subplot(224)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_motor[1], exp.pret:],nplt), label='Left',marker=',',color=exp.motor_color)
    plt.plot(timex_down_sampled,down_sample(r[exp.id_motor[2], exp.pret:],nplt), label='Right',marker=',',color=exp.motor_dark_color)
    if rt>0:
        plt.axvline(endt*exp.dt,color='black',linestyle='dashed',marker=',')
    plt.legend()
    plt.title('Motor')
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    plt.savefig(f'run_{nplt}_sample')
    
    fig=plt.figure(figsize=(5, 5))
    # nplt=1000
    # trj1=down_sample(r[exp.id_decision[1],exp.pret:],nplt)
    # trj2=down_sample(r[exp.id_decision[2],exp.pret:],nplt)
    k=np.ones(200)/200
    trj1=np.convolve(r[exp.id_decision[1]],k,'same')[exp.pret:-200]
    trj2=np.convolve(r[exp.id_decision[2]],k,'same')[exp.pret:-200]
    plt.scatter(trj1,trj2,marker=',')
    plt.plot(trj1[0],trj2[0],color='red',markersize=16,label='Start',linestyle='',zorder=4)
    plt.plot(trj1[endt],trj2[endt],color='purple',markersize=16,label='End',linestyle='',zorder=4)
    plt.plot([0,30],[0,30],color='black',linestyle='dashed',marker=',')
    plt.legend()
    plt.axis('square')
    plt.xlim(0,30)
    plt.ylim(0,30)

    plt.ylabel('Right decision module firing rate')
    plt.xlabel('Left decision module firing rate')
    plt.savefig('dec_trj')
    
    plt.show()


def down_sample(x,nsamp):
    sc=int(x.shape[-1]/nsamp)
    y=np.zeros(x.shape[:-1]+(nsamp,))
    for i in range(nsamp):
        y[...,i]=np.mean(x[...,i*sc:(i+1)*sc],axis=-1)
    return y

def vs_quantile(x,y,nbin=10,discard_extremes=False):
    
    b=np.quantile(x,[i/nbin for i in range(nbin+1)])
    arr = [y[(x < b[i]) * (x >= b[i-1])] for i in range(1, b.shape[0])]
    m=np.quantile(x,[(i+0.5)/nbin for i in range(nbin)])
    if discard_extremes:
        arr=arr[1:-1]
        m=m[1:-1]
    return m,arr
    

def init_with_args():
    
    import os
    import sys
    file=os.path.basename(sys.argv[0])
    

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("-m","--model",default="closed-loop")
    parser.add_argument("-s","--setting",default="normal")
    parser.add_argument("-e","--eval",action='store_true')
    
    args=parser.parse_args()
    
    
    if file!='experiment.py':
        if args.setting=="normal":
            args.setting=file.split('.')[0]
        else:
            args.setting=file.split('.')[0]+' '+args.setting
    
    if file=='compare.py':
        args.setting='compare'
    
    if not os.path.isdir(f'./{args.model}'):
        os.mkdir(f'./{args.model}')
    os.chdir(f'./{args.model}')

    if not os.path.isdir(f'./{args.setting}'):
        os.mkdir(f'./{args.setting}')
    os.chdir(f'./{args.setting}')

    return args.model,args.setting,args.eval

def run(experiment_class,analyzer_class,loop=False):
    
    model,setting,eval_only=init_with_args()
    print(f'Model: {model} Setting: {setting}')
    exp=experiment_class(model=model,setting=setting)

    if loop:
        while(1):
            r,t=exp.run_one(input_coh=0.032)
            show_run(exp,r,t)

    if not eval_only:
        print(f'Running experiments')
        exp.run_experiment()

    print(f'Analyzing data')
    a=analyzer_class(model=model,setting=setting)
    a.analyze_all()
