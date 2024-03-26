
import numpy as np
import matplotlib.pyplot as plt

from base import Base

from utils import *
from alter import Alter


class Analyzer(Base, Alter):
    """Basic class for data analysis"""

    def __init__(self, load_data=True, load_detailed_data=True, **kwargs):
        super(Analyzer, self).__init__(load_data=load_data,
                                       load_detailed_data=load_detailed_data, **kwargs)

        self.decision_aligned_delay = 100
        self.prepare()

    def analyze_all(self):
        self.draw_behavior()
        self.basic_analysis()
        self.COM_analysis(self.COM_based_on())
        

    def prepare(self):
        self.choice=self.trial_data[...,0]
        self.correct=(self.trial_data[...,0] == 2).astype(float)
        self.raw_rt=self.trial_data[...,1]
        self.reaction_time=(self.trial_data[...,1]-self.pret)*self.dt
        self.confidence = self.trial_data[...,2]
        self.confidence[self.confidence<0]=0  ## make confidence non-negative
        self.accunc = self.trial_data[...,3]
        self.difmot = self.trial_data[...,4]
        self.insunc = self.trial_data[...,5]

        self.miss=self.trial_data[..., 0] == 0
        self.rt_idx=(self.trial_data[...,1]*self.sample_rate).astype(int)
        
    def basic_analysis(self):



        for c in range(self.coh_level):
            print('  ·count of miss trials', np.sum(self.miss[c]))

        
        print('Accuracy: ', np.mean(self.correct, -1))
        print('Reaction time: ', np.mean(self.reaction_time, -1))

        def draw_ave(data,idx=None,name='ave'):
            fig=plt.figure(figsize=(15, 10))
            pret_idx = int(self.pret*self.sample_rate)
            data=data[...,pret_idx:]
            time = np.array(range(data.shape[-1]))/self.sample_rate*self.dt
            for c in range(self.coh_level):
                
                if idx is not None:
                    dat=data[c][idx[c]]
                else:
                    dat=data[c]
                    
                from utils import draw_errorbar
                plt.subplot(321)
                arr=[dat[:,self.id_decision[1],t] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=time,marker=',')
                plt.xlabel('Time')
                plt.ylabel('Dec incorrect')
                
                plt.subplot(322)
                arr=[dat[:,self.id_decision[2],t] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=time,marker=',')
                plt.xlabel('Time')
                plt.ylabel('Dec correct')
                
                plt.subplot(323)
                arr=[dat[:,self.id_motor[1],t] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=time,marker=',')
                plt.xlabel('Time')
                plt.ylabel('Mot incorrect')
                
                plt.subplot(324)
                arr=[dat[:,self.id_motor[2],t] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=time,marker=',')
                plt.xlabel('Time')
                plt.ylabel('Mot correct')

                plt.subplot(313)
                arr=[dat[:,self.id_unc[0],t] for t in range(dat.shape[-1])]
                draw_errorbar(arr,x=time,label=f'Coh {self.coh_list[c]}',marker=',')
                plt.xlabel('Time')
                plt.ylabel('Uncertainty')
                plt.legend()
            fig.savefig(f'{name}')
            return fig
        
        draw_ave(self.detailed_trial_data,~self.miss)

        plt.figure()

        for c in range(self.coh_level):
            plt.hist(self.reaction_time[c], 20, label=f'coherence {self.coh_list[c]}', alpha=0.7)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.savefig('reaction_time_hist')
    
        plt.figure()
        _,arr  = vs_quantile(self.confidence[~self.miss].flatten(),1.0*self.correct[~self.miss].flatten())

        draw_errorbar(arr)

        plt.xlabel('Confidence quantile')
        plt.ylabel('Accuracy')
        plt.savefig('acc_vs_conf')

        plt.figure()
        _,arr  = vs_quantile(self.accunc[~self.miss].flatten(),1.0*self.correct[~self.miss].flatten())

        draw_errorbar(arr)

        plt.xlabel('Uncertainty quantile')
        plt.ylabel('Accuracy')
        plt.savefig('acc_vs_accunc')

        plt.figure()
        arr = [self.difmot[coh][~self.miss[coh]]for coh in range(self.coh_level)]
        draw_errorbar(arr, label='All',color='steelblue')
        arr = [self.difmot[coh][self.choice[coh]==2] for coh in range(self.coh_level)]
        draw_errorbar(arr, label='Correct',color='seagreen')
        arr = [self.difmot[coh][self.choice[coh]==1] for coh in range(self.coh_level)]
        draw_errorbar(arr, label='Incorrect',color='firebrick')
        plt.legend()

        plt.xlabel('Coherence')
        plt.ylabel('Difmot')
        plt.savefig('difmot')

    def COM_based_on(self):
        return self.detailed_trial_data[:, :, self.id_motor[1]]-self.detailed_trial_data[:, :, self.id_motor[2]]

    def COM_analysis(self, data):


        def get_com(data,post_decision=False,b = 1,tb = 30):
            
            com = np.zeros((self.coh_level, self.repn))
            t_com = np.zeros((self.coh_level, self.nsamp))
            tb=self.sample_rate*tb/self.dt
            for c in range(self.coh_level):
                for r in range(self.repn):
                    l = 0
                    s = 0
                    if self.miss[c,r]:
                        continue
                    last_com = -1

                    if post_decision:
                        offset_idx=int((self.reaction_time[c,r]/self.dt+self.pret+self.delay)*self.sample_rate)
                        window=range(self.rt_idx[c,r], offset_idx)
                    else:
                        window=range(1, self.rt_idx[c,r])


                    for t in window:
                        if t>=data.shape[-1]:
                            break
                        if data[c, r, t]*data[c, r, t-1] <= 0:
                            l = t
                        if np.abs(data[c, r, t]) > b and s != np.sign(data[c, r, t]) and t >= l+tb:
                            if s:
                                last_com = t
                            s = np.sign(data[c, r, t])

                    if last_com != -1:

                        t_com[c, last_com] += 1

                        com[c, r] = last_com
                        if (int(data[c, r, last_com] > 0)):
                            com[c, r] = -last_com
            return com,t_com
        self.com,t_com=get_com(data)

        plt.figure()
        fig, ax = plt.subplots()
        ax.set_prop_cycle(self.res_cycler)
        draw_errorbar((self.com != 0)*1.0, label='All')
        draw_errorbar((self.com > 0)*1.0, label='Positive')
        draw_errorbar((self.com < 0)*1.0, label='Negative')
        plt.ylabel('Frequency')
        plt.xlabel('Coherence')
        plt.savefig('com')


        plt.figure()
        w = 30
        plt.figure(figsize=(18, 5))
        plt.subplot(121)
        plt.plot(np.convolve(np.mean(t_com, 0), np.ones(w), 'valid') / w)
        plt.ylabel('COM frequency')
        plt.xlabel('Time')
        plt.subplot(122)
        ave_com_t = np.array([np.sum([t_com[c, t]*t for t in range(t_com.shape[1])]
                                     )/np.sum(t_com[c]) for c in range(t_com.shape[0])])
        plt.plot(ave_com_t)
        plt.ylabel('COM time')
        plt.xlabel('coherence')
        plt.savefig('com_t')

        postd_data=data
        self.postd_com,t_com=get_com(postd_data,post_decision=True,b=0)
        
        plt.figure()
        fig, ax = plt.subplots()
        ax.set_prop_cycle(self.res_cycler)

        draw_errorbar((self.postd_com != 0)*1.0, label='All')
        draw_errorbar((self.postd_com > 0)*1.0, label='Positive')
        draw_errorbar((self.postd_com < 0)*1.0, label='Negative')

        # plt.legend()
        plt.ylabel('Frequency')
        plt.xlabel('Coherence')

        plt.savefig('postd_com')



    def get_decision_aligned_data(self):

        d=int(self.decision_aligned_delay*self.sample_rate)
        self.decision_aligned_detailed_trial_data = np.zeros_like(
            self.detailed_trial_data)
        for c in range(self.coh_level):
            for r in range(self.detailed_trial_data.shape[1]):
                rt = self.rt_idx[c,r]
                if rt <= 0:
                    rt = self.nsamp-1
                self.decision_aligned_detailed_trial_data[c, r] = np.concatenate(
                    (np.nan*self.detailed_trial_data[c, r][:, rt+1+d:], self.detailed_trial_data[c, r][:, :rt+1+d]), -1)

    def draw_behavior(self):

        from utils import draw_errorbar
        
        fig=plt.figure(figsize=(14, 7))
        axes = fig.subplots(2,2)
        for ax in axes.flatten():
            ax.set_prop_cycle(self.res_cycler)
            
        
        arr = self.correct
        draw_errorbar(arr,ax=axes[0][0])
        axes[0][0].set_xticks(range(self.trial_data.shape[0]))
        axes[0][0].set_ylabel('Accuracy')

        def draw_by_acc(data,ax):
            arr = [data[coh][~self.miss[coh]] for coh in range(self.coh_level)]
            draw_errorbar(arr,ax=ax, label='All')
            arr = [data[coh][self.choice[coh]==2] for coh in range(self.coh_level)]
            draw_errorbar(arr,ax=ax, label='Correct')
            arr = [data[coh][self.choice[coh]==1] for coh in range(self.coh_level)]
            draw_errorbar(arr,ax=ax, label='Incorrect')
            


        draw_by_acc(self.confidence,axes[0][1])
        axes[0][1].set_xticks(range(self.trial_data.shape[0]))
        axes[0][1].set_ylabel('Confidence')
        axes[0][1].legend()


        draw_by_acc(self.reaction_time,axes[1][0])
        axes[1][0].set_xticks(range(self.trial_data.shape[0]))
        axes[1][0].set_xlabel('Coherence level')
        axes[1][0].set_ylabel('Reaction time (ms)')


        draw_by_acc(self.accunc,axes[1][1])
        axes[1][1].set_xticks(range(self.trial_data.shape[0]))
        axes[1][1].set_xlabel('Coherence level')
        axes[1][1].set_ylabel('Uncertainty')

        fig.savefig('behavior')

        fig=plt.figure(figsize=(25, 4.8))
        axes = fig.subplots(1,4)
        for ax in axes.flatten():
            ax.set_prop_cycle(self.res_cycler)

        
        arr = self.correct
        draw_errorbar(arr,ax=axes[0], label='Motor')
        axes[0].set_xlabel('Coherence level')
        axes[0].set_ylabel('Accuracy')


        draw_by_acc(self.reaction_time,axes[1])
        axes[1].set_xlabel('Coherence level')
        axes[1].set_ylabel('Reaction time (ms)')


        draw_by_acc(self.confidence,axes[2])
        axes[2].set_xlabel('Coherence level')
        axes[2].set_ylabel('Confidence')

        draw_by_acc(self.accunc,axes[3])
        axes[3].set_xlabel('Coherence level')
        axes[3].set_ylabel('Uncertainty')
        axes[3].legend()

        fig.savefig('behavior_flat')


if __name__ == '__main__':
    pass
