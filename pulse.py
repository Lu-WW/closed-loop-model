

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
from tqdm import trange,tqdm

from experiment import Experiment

class PulseExperiment(Experiment):
    """Class for running Pulse experiments i.e. stimuli show changed coherence when the difference of motor modules reach the threshold"""
    def __init__(self,**kwargs):
        super(PulseExperiment, self).__init__(**kwargs)
        self.coh_list=[0,0.016,0.032,0.064]
        self.coh_level=len(self.coh_list)
        
        ## initialize basic pulse settings
        self.pulse_coh=0.128    
        self.pulse=0
        self.pulse_length=100
        self.pre_pulse_time=50
        self.pulse_thr_low=2
        self.pulse_thr_high=4
        
        self.wl_thr=int(1/self.dt)
        self.pulse_type={0:'no_pulse',1:'positive_pulse',-1:'negative_pulse'}


        
        if self.model.find('urg-on-dec')>=0:
            self.get_DV=self.get_DV_by_dec
            
            self.pulse_thr_low=4
            self.pulse_thr_high=6
            
            self.pulse_length=200


        if self.model.find('no-urg')>=0:
            self.get_DV=self.get_DV_by_dec
            self.pulse_thr_low=4
            self.pulse_thr_high=6
            
            self.pulse_length=200

        self.trial_data_name=f'{self.pulse_type[self.pulse]}_trial_data.npy'
        self.detailed_trial_data_name=f'{self.pulse_type[self.pulse]}_detailed_trial_data.npy'
            
        self.id_pulse_info=self.n_pop
        self.n_pop+=1
    def set_pulse(self,p):
        self.pulse=p
        self.trial_data_name=f'{self.pulse_type[self.pulse]}_trial_data.npy'
        self.detailed_trial_data_name=f'{self.pulse_type[self.pulse]}_detailed_trial_data.npy'

    def init_sti(self,input_coh,noise,r):
        input1 = self.input0*(1-input_coh)
        input2 = self.input0*(1+input_coh)

        for t in range(1, self.maxt):
            r[self.id_sti[1],t]=input1+self.input_noise*noise[self.id_sti[1],t-1]+self.input_baseline
            r[self.id_sti[2],t]=input2+self.input_noise*noise[self.id_sti[2],t-1]+self.input_baseline

        r[self.id_pulse_info,0]=np.random.rand()*(self.pulse_thr_high-self.pulse_thr_low)+self.pulse_thr_low # pulse threshold
        r[self.id_pulse_info,1]=self.maxt+1000/self.dt # pulse time

    def get_DV(self,r,t=None):
        '''
        get decision variable (based on motor modules)
        '''
        if t:
            return np.mean(r[...,self.id_motor[2],t-1-self.wl_thr:t]-r[...,self.id_motor[1],t-1-self.wl_thr:t],-1)
        else:
            return r[...,self.id_motor[2],:]-r[...,self.id_motor[1],:]
    def get_input(self,r,I_net,t):
        '''
        Apply pulse to the input
        '''
        pulset=r[self.id_pulse_info,1]
        thr=r[self.id_pulse_info,0]

        ## Start pulse
        if t-1>self.pret+self.pre_pulse_time/self.dt and pulset>self.maxt and np.abs(self.get_DV(r,t))>thr:
            pulset=t
            r[self.id_pulse_info,1]=pulset

        ## Apply pulse
        if t>=pulset and t<pulset+self.pulse_length/self.dt:
            r[self.id_sti[1],t] -= self.input0*(self.pulse_coh*self.pulse)
            r[self.id_sti[2],t] += self.input0*(self.pulse_coh*self.pulse)

            
        ## End pulse
        if t>=pulset+self.pulse_length/self.dt:
            r[self.id_sti[1],t]=self.input0
            r[self.id_sti[2],t]=self.input0
            # r[self.id_sti[1],t]=0
            # r[self.id_sti[2],t]=0
            pass
            
        I_net[self.id_decision[1]] += r[self.id_sti[1],t]
        I_net[self.id_decision[2]] += r[self.id_sti[2],t]


    def get_other_info(self,detailed_trial_data,r,coh,rep):
        '''
        help save data
        '''
        detailed_trial_data[coh,rep,self.id_pulse_info]=r[self.id_pulse_info,:self.nsamp]
        


            


from analyzer import Analyzer       
class PulseAnalyzer(Analyzer,PulseExperiment):
    """Class for data analysis of pulse experiments"""
    def __init__(self,**kwargs):
        super(PulseAnalyzer, self).__init__(load_data=False,load_detailed_data=False,**kwargs)
        ## positive and negative pulse
        self.pa=Analyzer(model=self.model,setting=self.setting,load_data=False,load_detailed_data=False)
        self.na=Analyzer(model=self.model,setting=self.setting,load_data=False,load_detailed_data=False)

        self.pa.trial_data=np.load('positive_pulse_trial_data.npy')
        self.pa.detailed_trial_data=np.load('positive_pulse_detailed_trial_data.npy')

        self.na.trial_data=np.load('negative_pulse_trial_data.npy')
        self.na.detailed_trial_data=np.load('negative_pulse_detailed_trial_data.npy')

        self.pa.trial_data=self.pa.trial_data[:self.coh_level]
        self.pa.detailed_trial_data=self.pa.detailed_trial_data[:self.coh_level]

        self.na.trial_data=self.na.trial_data[:self.coh_level]
        self.na.detailed_trial_data=self.na.detailed_trial_data[:self.coh_level]
        
        
        self.pa.prepare()
        self.na.prepare()
        

    def get_DV(self,r,t=None):
        if t:
            return np.mean(r[...,self.id_motor[2],t-1-self.wl_thr:t]-r[...,self.id_motor[1],t-1-self.wl_thr:t],-1)
        else:
            return r[...,self.id_motor[2],:]-r[...,self.id_motor[1],:]
    def analyze_all(self):

        
        self.draw_behavior()
        self.pulse_analysis()
        
    def draw_behavior(self):

        from utils import draw_errorbar
        fig=plt.figure(figsize=(20, 10))
        axes=fig.subplots(2,2)
        
        arr=self.na.correct
        draw_errorbar(arr,ax=axes[0][0],label='Negative pulse')
        arr=self.pa.correct
        draw_errorbar(arr,ax=axes[0][0],label='Positive pulse')
      
      
        axes[0][0].set_xlabel('Coherence level')
        axes[0][0].set_ylabel('Accuracy')
        axes[0][0].legend()



        arr=[self.na.confidence[c][~self.na.miss[c]] for c in range(self.na.confidence.shape[0])]
        draw_errorbar(arr,ax=axes[0][1],label='Negative pulse')
        arr=[self.pa.confidence[c][~self.pa.miss[c]] for c in range(self.pa.confidence.shape[0])]
        draw_errorbar(arr,ax=axes[0][1],label='Positive pulse')
        axes[0][1].set_xlabel('Coherence level')
        axes[0][1].set_ylabel('Confidence')

        
        

        arr=[self.na.reaction_time[c][~self.na.miss[c]] for c in range(self.na.reaction_time.shape[0])]
        draw_errorbar(arr,ax=axes[1][0],label='Negative pulse')
        arr=[self.pa.reaction_time[c][~self.pa.miss[c]] for c in range(self.pa.reaction_time.shape[0])]
        draw_errorbar(arr,ax=axes[1][0],label='Positive pulse')
        
        axes[1][0].set_xlabel('Coherence level')
        axes[1][0].set_ylabel('Reaction time (ms)')

        
       
        arr=[self.na.accunc[c][~self.na.miss[c]] for c in range(self.na.accunc.shape[0])]
        draw_errorbar(arr,ax=axes[1][1],label='Negative pulse')
        arr=[self.pa.accunc[c][~self.pa.miss[c]] for c in range(self.pa.accunc.shape[0])]
        draw_errorbar(arr,ax=axes[1][1],label='Positive pulse')
        
        axes[1][1].set_xlabel('Coherence level')
        axes[1][1].set_ylabel('Uncertainty')

        
        plt.savefig('behavior')

    def pulse_analysis(self):
        '''
        Conduct pulse-related analysis (e.g., see the effect of pulse across DV and time bins) and save figures
        '''
        negative_pulse_time_data=self.na.detailed_trial_data[:,:,self.id_pulse_info,1]
        positive_pulse_time_data=self.pa.detailed_trial_data[:,:,self.id_pulse_info,1]
        self.pulse_time=np.zeros((2,self.coh_level,self.repn))
        self.pulse_len=np.zeros((2,self.coh_level,self.repn))

        import copy
        self.positive_pulse_data=copy.deepcopy(self.pa.detailed_trial_data)
        self.negative_pulse_data=copy.deepcopy(self.na.detailed_trial_data)

        self.used=np.zeros((2,self.coh_level,self.repn),dtype=np.bool_)
        for c in range(self.coh_level):
            print('Pulse analysis: Coherence',self.coh_list[c])
            self.used[0,c]=(negative_pulse_time_data[c]<self.na.raw_rt[c])*(~self.na.miss[c])
            self.used[1,c]=(positive_pulse_time_data[c]<self.pa.raw_rt[c])*(~self.pa.miss[c])
            # self.used[0,c]=(negative_pulse_time_data[c]<self.na.raw_rt[c])*(~self.na.miss[c])*(negative_pulse_time_data[c]+self.pulse_length/self.dt<self.na.raw_rt[c])
            # self.used[1,c]=(positive_pulse_time_data[c]<self.pa.raw_rt[c])*(~self.pa.miss[c])*(positive_pulse_time_data[c]+self.pulse_length/self.dt<self.pa.raw_rt[c])
            
            
            print('  ·Count of discarded trials (Negative pulse)',self.repn-np.sum(self.used[0,c]),'/',self.repn)
            print('  ·Count of discarded trials (Positive pulse)',self.repn-np.sum(self.used[1,c]),'/',self.repn)

            for r in range(self.repn):
                if self.used[0,c,r]:
                    pulset=int((negative_pulse_time_data[c,r])/self.maxt*self.nsamp)-1
                    self.pulse_time[0,c,r]=(negative_pulse_time_data[c,r]-self.pret)*self.dt
                    self.negative_pulse_data[c,r]=np.concatenate((self.negative_pulse_data[c,r,:,pulset:],0*self.negative_pulse_data[c,r][:,:pulset]),-1)
                    self.pulse_len[0,c,r]=((self.na.raw_rt[c,r]-negative_pulse_time_data[c,r]))*self.dt
                    # self.pulse_len[0,c,r]=np.minimum(self.pulse_length,self.pulse_len[0,c,r])
                    
        

                if self.used[1,c,r]:
                    pulset=int((positive_pulse_time_data[c,r])/self.maxt*self.nsamp)-1
                    self.pulse_time[1,c,r]=(positive_pulse_time_data[c,r]-self.pret)*self.dt
                    self.positive_pulse_data[c,r]=np.concatenate((self.positive_pulse_data[c,r,:,pulset:],0*self.positive_pulse_data[c,r][:,:pulset]),-1)
                    self.pulse_len[1,c,r]=((self.pa.raw_rt[c,r]-positive_pulse_time_data[c,r]))*self.dt
                    # self.pulse_len[1,c,r]=np.minimum(self.pulse_length,self.pulse_len[1,c,r])
        

        

        negative_dv=self.get_DV(self.negative_pulse_data)
        positive_dv=self.get_DV(self.positive_pulse_data)

        #Show wt ms 
        wt=1000
        tl=int(wt/self.dt/self.maxt*self.nsamp)
        pulse_aligned_t=np.array(range(tl))/tl*wt
        

        self.positive_pulse_data=self.positive_pulse_data[:,:,:,:tl]
        self.negative_pulse_data=self.negative_pulse_data[:,:,:,:tl]
        
        baseline=np.mean((self.positive_pulse_data+self.negative_pulse_data)/2,1,keepdims=True)
        self.positive_pulse_data-=baseline
        self.negative_pulse_data-=baseline
        
        from utils import draw_errorbar
        plt.figure()
        arr=[self.pulse_time[0,c][self.used[0,c]] for c in range(self.coh_level)]
        draw_errorbar(arr,label='negative pulse')
        arr=[self.pulse_time[1,c][self.used[1,c]] for c in range(self.coh_level)]
        draw_errorbar(arr,label='positive pulse')
        plt.xlabel('Coherence level')
        plt.ylabel('Time')
        plt.legend()
        plt.savefig('pulse_time')

        fig=plt.figure(figsize=(15,10))
        axes=fig.subplots(3,2)
        for c in range(self.coh_level):
            dat=self.negative_pulse_data[c][self.used[0,c]]
            arr=[dat[:,self.id_decision[1],t]-dat[:,self.id_decision[2],t] for t in range(tl)]
            draw_errorbar(arr,x=pulse_aligned_t,ax=axes[0][0],label=f'coherence {self.coh_list[c]}')

            dat=self.positive_pulse_data[c][self.used[1,c]]
            arr=[dat[:,self.id_decision[1],t]-dat[:,self.id_decision[2],t] for t in range(tl)]
            draw_errorbar(arr,x=pulse_aligned_t,ax=axes[0][1],label=f'coherence {self.coh_list[c]}')


            dat=self.negative_pulse_data[c][self.used[0,c]]
            arr=[dat[:,self.id_motor[2],t]-dat[:,self.id_motor[1],t] for t in range(tl)]
            draw_errorbar(arr,x=pulse_aligned_t,ax=axes[1][0],label=f'coherence {self.coh_list[c]}')

            dat=self.positive_pulse_data[c][self.used[1,c]]
            arr=[dat[:,self.id_motor[2],t]-dat[:,self.id_motor[1],t] for t in range(tl)]
            draw_errorbar(arr,x=pulse_aligned_t,ax=axes[1][1],label=f'coherence {self.coh_list[c]}')


            dat=self.negative_pulse_data[c][self.used[0,c]]
            arr=[dat[:,self.id_unc[0],t] for t in range(tl)]
            draw_errorbar(arr,x=pulse_aligned_t,ax=axes[2][0],label=f'coherence {self.coh_list[c]}')

            dat=self.positive_pulse_data[c][self.used[1,c]]
            arr=[dat[:,self.id_unc[0],t] for t in range(tl)]
            draw_errorbar(arr,x=pulse_aligned_t,ax=axes[2][1],label=f'coherence {self.coh_list[c]}')


        axes[0][0].set_xlabel('time')
        axes[0][0].set_ylabel(r'$\Delta$ dec')
        axes[0][1].set_xlabel('Time')
        axes[0][1].set_ylabel(r'$\Delta$ dec')

        axes[1][0].set_xlabel('Time')
        axes[1][0].set_ylabel(r'$\Delta$ mot')
        axes[1][1].set_xlabel('Time')
        axes[1][1].set_ylabel(r'$\Delta$ mot')

        axes[2][0].set_xlabel('Time')
        axes[2][0].set_ylabel(r'$\Delta$ unc')
        axes[2][1].set_xlabel('Time')
        axes[2][1].set_ylabel(r'$\Delta$ unc')


        axes[0][0].set_title('negative pulse')
        axes[0][1].set_title('positive pulse')

        axes[0][0].legend()
        for ax in axes.flatten():
            ax.axvline(self.pulse_length,color='black',linestyle='dashed')
        fig.savefig('pulse_aligned')


        fig=plt.figure(figsize=(15,10))
        axes=fig.subplots(3,1)
        for c in range(self.coh_level):
            dat=np.mean(self.positive_pulse_data[c][self.used[1,c]],0)-np.mean(self.negative_pulse_data[c][self.used[0,c]],0)
            axes[0].plot(pulse_aligned_t,dat[self.id_decision[2]]-dat[self.id_decision[1]],label=f'coherence {self.coh_list[c]}')
            axes[1].plot(pulse_aligned_t,dat[self.id_motor[2]]-dat[self.id_motor[1]],label=f'coherence {self.coh_list[c]}')
            axes[2].plot(pulse_aligned_t,dat[self.id_unc[0]],label=f'coherence {self.coh_list[c]}')


           


    
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel(r'$\Delta$ dec')

        axes[1].set_xlabel('Time')
        axes[1].set_ylabel(r'$\Delta$ mot')

        axes[2].set_xlabel('Time')
        axes[2].set_ylabel(r'$\Delta$ unc')

        axes[0].legend()

        
        for ax in axes.flatten():
            ax.axvline(self.pulse_length,color='black',linestyle='dashed')
        fig.savefig('pulse_aligned_comp')


        begint=int(0/self.dt/self.maxt*self.nsamp)+1

        self.base_dv=np.zeros_like(self.pulse_time)
        
        self.base_dv[0]=negative_dv[:,:,0]
        self.base_dv[1]=positive_dv[:,:,0]

        def init():
            
            self.negative_correct=copy.deepcopy(self.na.correct)
            self.positive_correct=copy.deepcopy(self.pa.correct)

            self.delt_dv=np.zeros((2,self.coh_level,self.repn))
            
            for c in range(self.coh_level):
                for r in range(self.repn):
                    
                    if self.used[0,c,r]:        
                        endt=int((self.na.raw_rt[c,r]-negative_pulse_time_data[c,r])/self.maxt*self.nsamp)+begint
                        # endt=int((self.pulse_length/self.dt)/self.maxt*self.nsamp)+begint
                        # self.delt_dv[0,c,r]=negative_dv[c,r,endt]-negative_dv[c,r,begint]
                        self.delt_dv[0,c,r]=negative_dv[c,r,endt]-self.base_dv[0,c,r]
                        


                    if self.used[1,c,r]:
                        
                        endt=int((self.pa.raw_rt[c,r]-positive_pulse_time_data[c,r])/self.maxt*self.nsamp)+begint
                        # endt=int((self.pulse_length/self.dt)/self.maxt*self.nsamp)+begint
                        # self.delt_dv[1,c,r]=positive_dv[c,r,endt]-positive_dv[c,r,begint]
                        self.delt_dv[1,c,r]=positive_dv[c,r,endt]-self.base_dv[1,c,r]

    
        def analyze(to_exclude,to_analyze,name_analyze):
            '''
            To control the effect of the 'to_exclude'
            '''
            nq=10
            b=np.quantile(to_exclude[self.used],[i/nq for i in range(nq+1)])
            
            for i in range(1,b.shape[0]):
                for c in range(self.coh_level):
                    idx=(to_exclude[0,c]<b[i])*(to_exclude[0,c]>=b[i-1])*self.used[0,c]
                    self.delt_dv[0,c,idx]-=np.mean(self.delt_dv[0,c,idx])
                    self.negative_correct[c,idx]-=np.mean(self.negative_correct[c,idx])


                    idx=(to_exclude[1,c]<b[i])*(to_exclude[1,c]>=b[i-1])*self.used[1,c]
                    self.delt_dv[1,c,idx]-=np.mean(self.delt_dv[1,c,idx])
                    self.positive_correct[c,idx]-=np.mean(self.positive_correct[c,idx])
                    
                               

            nq=10
            b=np.quantile(to_analyze[self.used].flatten(),[i/nq for i in range(nq+1)])
            
            from utils import draw_bootstrap

            arr1=[self.positive_correct[(to_analyze[1]<b[i])*(to_analyze[1]>=b[i-1])*self.used[1]] for i in range(1,b.shape[0])]
            arr2=[self.negative_correct[(to_analyze[0]<b[i])*(to_analyze[0]>=b[i-1])*self.used[0]] for i in range(1,b.shape[0])]


            plt.figure()
            y_acc,y_bar_acc=draw_bootstrap(arr1,arr2,x=(b[1:]+b[:-1])/2,label='All')
            draw_errorbar(arr1,x=(b[1:]+b[:-1])/2,label='Positive')
            draw_errorbar(arr2,x=(b[1:]+b[:-1])/2,label='Negative')
            plt.axhline(0,color='black',linestyle='dashed',marker=',')
            plt.legend()
            plt.ylabel(r'residual $\Delta$choice')
            plt.xlabel(name_analyze)
            plt.savefig(f'change_of_accuracy_by_{name_analyze}')
            plt.close()
            
           
            arr1=[self.delt_dv[1][(to_analyze[1]<b[i])*(to_analyze[1]>=b[i-1])*self.used[1]] for i in range(1,b.shape[0])]
            arr2=[self.delt_dv[0][(to_analyze[0]<b[i])*(to_analyze[0]>=b[i-1])*self.used[0]] for i in range(1,b.shape[0])]
            
            plt.figure()
            y_dv,y_bar_dv=draw_bootstrap(arr1,arr2,x=(b[1:]+b[:-1])/2,label='All')
            draw_errorbar(arr1,x=(b[1:]+b[:-1])/2,label='Positive')
            draw_errorbar(arr2,x=(b[1:]+b[:-1])/2,label='Negative')
            plt.axhline(0,color='black',linestyle='dashed',marker=',')
            plt.legend()
            plt.ylabel(r'residual $\Delta$DV')
            plt.xlabel(name_analyze)
            plt.savefig(f'change_of_dv_by_{name_analyze}')
            plt.close()
            
            
            return ((b[1:]+b[:-1])/2,(y_acc,y_bar_acc),(y_dv,y_bar_dv))

        
        init()
        
        
        

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        for c in range(self.coh_level):
            plt.hist(self.pulse_time[0,c][self.used[0,c]].flatten(),100)
        plt.ylabel('Count')
        plt.xlabel('Pulse time')
        plt.title('Negative pulse')
        plt.subplot(122)
        for c in range(self.coh_level):
            plt.hist(self.pulse_time[1,c][self.used[1,c]].flatten(),100,label=f'Coh level {c}')
        plt.title('Positive pulse')
        plt.legend()
        plt.xlabel('Pulse time')
        
        plt.savefig('pulse_time_hist')


        plt.figure()
        
        from utils import vs_quantile
        nq=10
        
        b=np.quantile(self.base_dv[self.used],[i/nq for i in range(nq+1)])
            
        pl=copy.deepcopy(self.pulse_len)
        for i in range(1,b.shape[0]):
            for c in range(self.coh_level):
                idx=(self.base_dv[0,c]<b[i])*(self.base_dv[0,c]>=b[i-1])*self.used[0,c]
                pl[0,c,idx]-=np.mean(pl[0,c,idx])
                idx=(self.base_dv[1,c]<b[i])*(self.base_dv[1,c]>=b[i-1])*self.used[1,c]
                pl[1,c,idx]-=np.mean(pl[1,c,idx])
                

        neg_x,arr=vs_quantile(self.pulse_time[0][self.used[0]],pl[0][self.used[0]])
        neg_y,neg_y_bar=draw_errorbar(arr,x=neg_x,label='Negative pulse')
        pos_x,arr=vs_quantile(self.pulse_time[1][self.used[1]],pl[1][self.used[1]])
        pos_y,pos_y_bar=draw_errorbar(arr,x=pos_x,label='Positive pulse')
        neg_pl=(neg_x,neg_y,neg_y_bar)
        pos_pl=(pos_x,pos_y,pos_y_bar)
        # plt.ylim(-200,700)
        plt.legend()
        plt.xlabel('Pulse time')
        plt.ylabel('Residual pulse length')
        plt.savefig('pulse_len_vs_pulse_time')

        plt.figure()
        x,arr=vs_quantile(self.pulse_time,self.base_dv,nbin=20)
        draw_errorbar(arr,x=x)
        plt.xlabel('Pulse time')
        plt.ylabel('DV at pulse start')
        plt.savefig('DV_vs_pulse_time')
        
                   
        plt.figure()
        
        mixed_x,arr=vs_quantile(self.pulse_time[self.used],pl[self.used])
        mixed_y,mixed_y_bar=draw_errorbar(arr,x=mixed_x)
        mixed_pl=(mixed_x,mixed_y,mixed_y_bar)
        
        # plt.ylim(-200,700)
        plt.xlabel('Pulse time')
        plt.ylabel('Residual pulse length')
        plt.savefig('mixed_pulse_len_vs_pulse_time')

        init()
        effect_by_time=analyze(self.base_dv,self.pulse_time,'time')
       
        init()   
        effect_by_dv=analyze(self.pulse_time,np.abs(self.base_dv),'dv')

        return effect_by_dv,effect_by_time,pos_pl,neg_pl,mixed_pl
        
     



if __name__=='__main__':

    from utils import init_with_args
    model,setting,eval_only=init_with_args()

    exp=PulseExperiment(model=model,setting=setting)
    

    if not eval_only:
        exp.set_pulse(1)
        exp.run_experiment()
        exp.set_pulse(-1)
        exp.run_experiment()

    a=PulseAnalyzer(model=model,setting=setting)
    a.analyze_all()

