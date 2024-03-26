
import os
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
from tqdm import trange,tqdm

from pulse import PulseExperiment,PulseAnalyzer

if __name__=='__main__':

    from utils import init_with_args
    model,setting,eval_only=init_with_args()

    exp=PulseExperiment(model=model,setting=setting)
    base_A_urg=exp.A_urg

    fig1=plt.figure(figsize=(20,12))
    axes1=fig1.subplots(2,2)
    fig2=plt.figure(figsize=(20,6))
    axes2=fig2.subplots(1,2)
    fig3=plt.figure(figsize=(20,6))
    axes3=fig3.subplots(1,1)
    high_urg_level=6
    # low_urg_level=0
    low_urg_level=2
    if model.find('no-urg')>=0:
        high_urg_level=3
    urgs=range(low_urg_level,high_urg_level,1)


    from cycler import cycler
    color_cycler=cycler('color', plt.cm.GnBu(np.linspace(0.4,0.8,len(urgs))))
    for ax in axes1.flatten():
        ax.set_prop_cycle(color_cycler) 
    for ax in axes2.flatten():
        ax.set_prop_cycle(color_cycler)
    axes3.set_prop_cycle(color_cycler)


    for i in urgs:
        exp.A_urg=base_A_urg*i
      
        if not os.path.isdir(f'./urgency level {i}'):
            os.mkdir(f'./urgency level {i}')
        os.chdir(f'./urgency level {i}')

        if not eval_only:
            print(f'Running urgency level {i}')
            exp.set_pulse(1)
            exp.run_experiment()
            exp.set_pulse(-1)
            exp.run_experiment()             
            # exp.set_pulse(0)
            # exp.run_experiment()                                                                                                 
        
        print(f'Analyzing urgency level {i}')
        a=PulseAnalyzer(model=model,setting=setting,exp_maxt=exp.exp_maxt)
        a.draw_behavior()
        effect_by_dv,effect_by_time,pos_pl,neg_pl,mixed_pl=a.pulse_analysis()



        x,(y_acc,y_bar_acc),(y_dv,y_bar_dv)=effect_by_dv
        x=np.array(range(x.shape[0]))


        l,=axes1[0][0].plot(x,y_acc,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes1[0][0].fill_between(x,y_acc-y_bar_acc,y_acc+y_bar_acc,alpha=0.2,color=l.get_color())


        l,=axes1[0][1].plot(x,y_dv,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes1[0][1].fill_between(x,y_dv-y_bar_dv,y_dv+y_bar_dv,alpha=0.2,color=l.get_color())


        x,(y_acc,y_bar_acc),(y_dv,y_bar_dv)=effect_by_time
        x=np.array(range(x.shape[0]))

        l,=axes1[1][0].plot(x,y_acc,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes1[1][0].fill_between(x,y_acc-y_bar_acc,y_acc+y_bar_acc,alpha=0.2,color=l.get_color())


        l,=axes1[1][1].plot(x,y_dv,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes1[1][1].fill_between(x,y_dv-y_bar_dv,y_dv+y_bar_dv,alpha=0.2,color=l.get_color())


        
        
        neg_x,neg_y,neg_y_bar=neg_pl
        pos_x,pos_y,pos_y_bar=pos_pl
        l,=axes2[0].plot(neg_x,neg_y,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes2[0].fill_between(neg_x,neg_y-neg_y_bar,neg_y+neg_y_bar,alpha=0.2,color=l.get_color())
        l,=axes2[1].plot(pos_x,pos_y,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes2[1].fill_between(pos_x,pos_y-pos_y_bar,pos_y+pos_y_bar,alpha=0.2,color=l.get_color())

        mixed_x,mixed_y,mixed_y_bar=mixed_pl
        l,=axes3.plot(mixed_x,mixed_y,label=r'$\alpha_{urg}=$'+'{:g}'.format(exp.A_urg),marker=',')
        axes3.fill_between(mixed_x,mixed_y-mixed_y_bar,mixed_y+mixed_y_bar,alpha=0.2,color=l.get_color())\

        os.chdir(f'..')


    axes1[0][0].set_xlabel('DV quantile')
    axes1[0][0].set_ylabel(r'residual $\Delta$choice')
    axes1[0][0].axhline(0,color='black',linestyle='dashed',marker=',')
    axes1[0][0].set_ylim(-0.2,0.3)

    axes1[0][1].set_xlabel('DV quantile')
    axes1[0][1].set_ylabel(r'residual $\Delta$DV')
    axes1[0][1].axhline(0,color='black',linestyle='dashed',marker=',')

    axes1[1][0].set_xlabel('pulse time quantile')
    axes1[1][0].set_ylabel(r'residual $\Delta$choice')
    axes1[1][0].axhline(0,color='black',linestyle='dashed',marker=',')
    axes1[1][0].set_ylim(-0.2,0.3)

    axes1[1][1].set_xlabel('pulse time quantile')
    axes1[1][1].set_ylabel(r'residual $\Delta$DV')
    axes1[1][1].axhline(0,color='black',linestyle='dashed',marker=',')

    axes2[0].axhline(0,color='black',linestyle='dashed',marker=',')
    axes2[0].set_ylim(-200,700)
    axes2[0].set_xlabel('Pulse time (ms)')
    axes2[0].set_ylabel('Residual pulse length (ms)')
    axes2[0].set_title('Negative pulse')

    axes2[1].axhline(0,color='black',linestyle='dashed',marker=',')
    axes2[1].set_ylim(-200,700)
    axes2[1].set_xlabel('Pulse time (ms)')
    axes2[1].set_ylabel('Residual pulse length (ms)')
    axes2[1].set_title('Positive pulse')

    
    axes3.axhline(0,color='black',linestyle='dashed',marker=',')
    axes3.set_ylim(-200,700)
    axes3.set_xlabel('Pulse time (ms)')
    axes3.set_ylabel('Residual pulse length (ms)')
    if high_urg_level>2:
        import matplotlib.colors as colors
        cmap=colors.ListedColormap(plt.cm.GnBu(np.linspace(0.4,0.8,256)))
        cax = fig1.add_axes([axes1[0][0].get_position().x1-0.22,axes1[0][0].get_position().y1-0.04,0.2,0.02])
        cb=fig1.colorbar(plt.cm.ScalarMappable(cmap=cmap),cax=cax,orientation='horizontal')
        cb.set_ticks([0,1],labels=[base_A_urg*(min(urgs)),base_A_urg*(max(urgs))])
        fig1.text(cax.get_position().x0+0.1,cax.get_position().y1+0.01,r'$\alpha_{urg}$',ha='center')
        cb.outline.set_visible(False)
        # axes1[0][0].legend()
        # axes2[0].legend()
        # axes3.legend()
    fig1.savefig('pulse_effect_by_urgency_level')
    fig2.savefig('pulse_len_vs_pulse_time')
    fig3.savefig('mixed_pulse_len_vs_pulse_time')

