

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
from tqdm import trange,tqdm

from experiment import Experiment
from analyzer import Analyzer

class AttentionExperiment(Experiment):
    """class for attention experiments"""
    def __init__(self,**kwargs):
        super(AttentionExperiment, self).__init__(**kwargs)
        

        self.switch_constant=1/300
        self.switch_coef=0.1
        self.switch_noise=1/100
        self.attention_modulation='input'
        self.attention_bias=1.2
        if self.setting.find('dec')>=0:
            self.attention_modulation='decision'
            self.attention_bias=0.01
        if self.setting.find('mot')>=0:
            self.attention_modulation='motor'
            self.attention_bias=0.01

        self.id_att_status=self.n_pop
        self.n_pop+=1

        self.id_att_info=self.n_pop
        self.n_pop+=1
        self.id_att_var=self.n_pop
        self.n_pop+=1

    def init_sti(self,input_coh,noise,r):
        input1 = self.input0*(1-input_coh)
        input2 = self.input0*(1+input_coh)

        for t in range(1, self.maxt):
            r[self.id_sti[1],t]=input1+self.input_noise*noise[self.id_sti[1],t-1]+self.input_baseline
            r[self.id_sti[2],t]=input2+self.input_noise*noise[self.id_sti[2],t-1]+self.input_baseline


        r[self.id_att_var,:]=0
        r[self.id_att_info,:]=0
        r[self.id_att_status,:]=int(np.random.rand()>0.5)+1

        r[self.id_att_info,0]=1
        r[self.id_att_info,1]=self.pret
        if r[self.id_att_status,0]==1:
            r[self.id_att_info,1]=-self.pret

        

    def get_input(self,r,I_net,t):
        delt=(self.switch_constant/(1+self.switch_coef*r[self.id_accunc[0],t-1])+np.random.randn()*self.switch_noise/np.sqrt(self.dt))*self.dt
        r[self.id_att_var,t]=delt
        if (r[self.id_att_var,t-1]<1):
            r[self.id_att_var,t]+=r[self.id_att_var,t-1]
        r[self.id_att_status,t]=r[self.id_att_status,t-1]

        if r[self.id_att_var,t]>=1:
          r[self.id_att_status,t]=3-r[self.id_att_status,t]

          r[self.id_att_info,0]+=1
          cnt=int(r[self.id_att_info,0])
          r[self.id_att_info,cnt]=t
          if r[self.id_att_status,t]==1:
              r[self.id_att_info,cnt]=-t

        attention=int(r[self.id_att_status,t])
        if self.attention_modulation=='input':
            r[self.id_sti[attention],t]*=self.attention_bias
        elif self.attention_modulation=='decision':
            I_net[self.id_decision[attention]]+=self.attention_bias
            # r[self.id_sti[attention],t]+=self.attention_bias
        elif self.attention_modulation=='motor':
            I_net[self.id_motor[attention]]+=self.attention_bias

        
        I_net[self.id_decision[1]] += r[self.id_sti[1],t]
        I_net[self.id_decision[2]] += r[self.id_sti[2],t]


    def get_other_info(self,detailed_trial_data,r,coh,rep):
        detailed_trial_data[coh,rep,self.id_att_info]=r[self.id_att_info,:self.nsamp]
        


        

    

class AttentionAnalyzer(Analyzer,AttentionExperiment):
    """Class for data analysis of attetion experiments"""
    def __init__(self,**kwargs):
        super(AttentionAnalyzer, self).__init__(load_data=True,load_detailed_data=True,**kwargs)
     
    def analyze_all(self):
        self.prepare()
        self.draw_behavior()
        self.attention_analysis()

    def get_att_list(self,c,r):
        rt=self.raw_rt[c,r]
        pret=int(self.pret*self.sample_rate)
        if rt==0:
            rt=self.nsamp
        data=self.detailed_trial_data[c,r]
        
        l=[]
        
        for i in range(1,1+int(data[self.id_att_info,0])):
            if data[self.id_att_info,i]>rt:
                break

            if data[self.id_att_info,i]>0:
                s=2
            else:
                s=1

            t=(np.abs(data[self.id_att_info,i])-self.pret)*self.dt
            l.append((t,s))
                

        return l
    def attention_analysis(self,att_list=None,legend=True):
        
        cnt_att=np.zeros((self.coh_level,self.repn))
        last_att=np.zeros((self.coh_level,self.repn))
        first_att=np.zeros((self.coh_level,self.repn))
        len_att=np.zeros((self.coh_level,self.repn,3))
        
        ## Pseudo left or right for visualization
        np.random.seed(100)
        idx=np.random.rand(self.coh_level,self.repn)>0.5
        idx[0]=True
        def get_directed(data1,data2,condition1=None,condition2=None):
            ret=[]
            for c in range(-self.coh_level+1,self.coh_level):
                if c>=0:
                    if condition1 is None:
                        cond=~self.miss[np.abs(c)]
                    else:
                        cond=(~self.miss[np.abs(c)])*condition1[np.abs(c)]
                    item=data1[np.abs(c),cond*(idx[np.abs(c)]==True)]
                else:
                    if condition2 is None:
                        cond=~self.miss[np.abs(c)]
                    else:
                        cond=(~self.miss[np.abs(c)])*condition2[np.abs(c)]
                    item=data2[np.abs(c),cond*(idx[np.abs(c)]==False)]
                    
                ret.append(item)
            return ret
            
        mid_len=[[] for c in range(self.coh_level)]
        first_len=[[] for c in range(self.coh_level)]
        last_len=[[] for c in range(self.coh_level)]
        for c in range(self.coh_level):
            for r in range(self.repn):
                if att_list is None:
                    l=self.get_att_list(c,r)
                else:
                    l=att_list[c][r]
                for i in range(1,len(l)):
                    d=l[i][0]-l[i-1][0]
                    len_att[c,r,l[i-1][1]]+=d
                    if i>1:
                        mid_len[c].append(d)
                    if i==1:
                        first_len[c].append(d)
                        
                if not self.miss[c,r]:
                    d=self.reaction_time[c,r]-l[-1][0]
                else:
                    d=self.exp_maxt-l[-1][0]
                len_att[c,r,l[-1][1]]+=d
                
                if len(l)==1:
                    first_len[c].append(d)
                last_len[c].append(d)

                first_att[c][r]=l[0][1] 
                last_att[c][r]=l[-1][1] 
                cnt_att[c][r]=len(l)
                
        
        plt.figure(figsize=(8,4.8))
        from utils import draw_errorbar
        att1=[last_att[c]==1 for c in range(self.coh_level)]
        att2=[last_att[c]==2 for c in range(self.coh_level)]
        arr=get_directed(self.choice==2,self.choice==1)
        draw_errorbar(arr,label='All',color='deepskyblue')
        arr=get_directed(self.choice==2,self.choice==1,att2,att1)
        draw_errorbar(arr,label='Last left',color=self.decision_color)
        arr=get_directed(self.choice==2,self.choice==1,att1,att2)
        draw_errorbar(arr,label='Last right',color=self.decision_dark_color)
        plt.axhline(0.5,color='black',linestyle='dashed',marker=',')
        plt.xlabel('Left$\minus$Right level')
        plt.ylabel('Proportion left')
        plt.xticks(range(2*self.coh_level-1),range(-self.coh_level+1,self.coh_level))
        if legend:
            plt.legend()
        plt.savefig('last_att')

        plt.figure(figsize=(8,4.8))
        plt.plot(np.mean(cnt_att,1))
        # arr=[(cnt_att[c,directed_idx(c)]/exp.trial_data[c,directed_idx(c),3])[exp.trial_data[c,directed_idx(c),3]!=0] for c in range(self.coh_level)]
        # draw_errorbar(arr,label='Switch rate')
        plt.xlabel('Coherence')
        plt.ylabel('Attention count')
        plt.savefig('cnt_att')
        

        plt.figure(figsize=(8,4.8))
        draw_errorbar(first_len,label='First')
        draw_errorbar(mid_len,label='Middle')
        draw_errorbar(last_len,label='Last')
        plt.xlabel('Coherence level')
        plt.ylabel('Attention length (ms)')
        if legend:
            plt.legend()
        plt.savefig('att_len')

        plt.figure(figsize=(8,4.8))
        res2=(self.choice==2)*1.0
        res2-=np.mean(res2,0,keepdims=True)
        res1=(self.choice==1)*1.0
        res1-=np.mean(res1,0,keepdims=True)
       
        n_bin=6
        adv=len_att[:,:,2]-len_att[:,:,1]
        
        b=np.quantile(adv,[i/n_bin for i in range(n_bin+1)])
        
        l=[]
        for i in range(n_bin):
            att1=(len_att[:,:,2]-len_att[:,:,1]>=b[i])*(len_att[:,:,2]-len_att[:,:,1]<b[i+1])
            att2=(len_att[:,:,1]-len_att[:,:,2]>=b[i])*(len_att[:,:,1]-len_att[:,:,2]<b[i+1])
            arr=get_directed(res2,res1,att1,att2)
            li=[]
            
            for a in arr:
                li.extend(a)
            l.append(li)
        y,y_bar=draw_errorbar(l,x=(b[1:]+b[:-1])/2)
        plt.axhline(0,color='black',linestyle='dashed',marker=',')
        m=np.max(np.abs(y)+np.abs(y_bar))
        plt.ylim(-1.1*m,1.1*m)
        plt.xlabel('Left advantage time (ms)')
        
        plt.ylabel('Corrected proportion left')
        plt.savefig('adv_att')
        # plt.show()


        return first_len,mid_len,last_len


if __name__=='__main__':


    from utils import run
    run(AttentionExperiment,AttentionAnalyzer)
