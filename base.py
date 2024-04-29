

import numpy as np

import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings('ignore')
class Base():

    """Basic class"""

    def __init__(self,model='closed-loop',setting='normal',load_data=False,load_detailed_data=False,repn=10000,exp_maxt=3000,coh_list=[0,0.016,0.032,0.064,0.128,0.256,0.512]):
        # Auxiliary variables
        self.model=model
        self.setting=setting
        self.trial_data_name='trial_data.npy'
        self.detailed_trial_data_name='detailed_trial_data.npy'

        # Basic setting
        self.dt = 1
        self.pret=int(300/self.dt)
        self.thr = 30
        self.nchoice=2
        self.repn = repn
        self.exp_maxt=exp_maxt
        self.maxt = int(self.exp_maxt/self.dt)+self.pret
        self.sample_rate=1/30
        self.nsamp=int(self.maxt*self.dt*self.sample_rate)
        
        self.coh_list=coh_list
        self.coh_level = len(self.coh_list)


        # Phy parameters
        self.a = 270
        self.b = 108
        self.d = 0.154
        self.kappa=1

        # Dynamic parameters
        self.input0=1e-4*5.2*30
        self.I_0 = 0.3255
        self.tau_N = 100
        self.gamma = 0.641
        self.tau_noise = 2
        self.tau_r = 2
        self.sigma_noise = 0.02
        self.unc_add=-0.2
        self.tau_acc=100
        self.J_acc=0.5*self.tau_acc
        self.A_urg=1/500
        self.input_noise=0
        self.input_baseline=0
        self.delay=0

        # Plotting parameters
        self.stim_color='lightgray'
        self.decision_color='lightblue'
        self.motor_color='lightgreen'

        self.stim_dark_color='gray'
        self.decision_dark_color='steelblue'
        self.motor_dark_color='limegreen'

        self.accunc_color='orange'
        self.insunc_color='gold'

        from cycler import cycler
        self.res_cycler=cycler('color', ['steelblue','seagreen','firebrick'])
            
        # Initialize neural populations and connections
        self.init_populations()
        self.init_connections()

        # Initialize model and setting
        self.model=model
        self.setting=setting
        self.init_model()
        self.init_setting()
        
        # Load data
        if load_data:
            self.trial_data=np.load('trial_data.npy')
        else:
            self.trial_data=np.zeros((self.coh_level,self.repn,6))
        if load_detailed_data:
            self.detailed_trial_data=np.load('detailed_trial_data.npy')


    def init_populations(self):

        self.n_pop = 0 

        self.id_motor=np.zeros(self.nchoice+5,np.int32)
        self.id_decision=np.zeros(self.nchoice+5,np.int32)

        self.id_unc=np.zeros(self.nchoice+5,np.int32)
        self.id_sti=np.zeros(self.nchoice+5,np.int32)

        self.id_accunc=np.zeros(self.nchoice+5,np.int32)


        for c in range(1,1+self.nchoice):

            self.id_motor[c]=self.n_pop
            self.n_pop+=1
            self.id_decision[c]=self.n_pop
            self.n_pop+=1
        self.id_unc[0]=self.n_pop
        self.n_pop+=1

        self.id_accunc[0]=self.n_pop
        self.n_pop+=1
        
        self.n_dyn=self.n_pop #number of dynamic modules
         
        for c in range(1,1+self.nchoice):
            self.id_sti[c]=self.n_pop
            self.n_pop+=1




    def set_edge(self,x,y,w,s):
        self.W[x,y]=w
        if (s==0):
            self.W[x,y]=-w

    def init_connections(self):
        self.W = np.zeros((self.n_dyn, self.n_dyn))

        for i in range(self.n_dyn):
            # self.set_edge(i,i,0.2609,1)
            self.set_edge(i,i,0.2440182353,1)

        for c in range(1,1+self.nchoice):

            ###decision & motor

            for c2 in range(1,1+self.nchoice):
                if (c==c2):
                    continue

                # self.set_edge(self.id_decision[c],self.id_decision[c2],0.0497,0)
                self.set_edge(self.id_decision[c],self.id_decision[c2],0.005,0)

                
                self.set_edge(self.id_decision[c],self.id_motor[c2],0.3,0)

                self.set_edge(self.id_motor[c],self.id_decision[c2],0.02,0)



            self.set_edge(self.id_decision[c],self.id_motor[c],0.2,1)


            ###d&m to control

            
            self.set_edge(self.id_decision[c],self.id_unc[0],0.25,1)

            ###control to d&m


            self.set_edge(self.id_unc[0],self.id_decision[c],0.02,0)



    def Phy(self,I):

        x = self.a * I - self.b
        ret =  x / (1 - np.exp(-self.d * x))

        return ret

    def Phy_acc(self,I):
        ret=self.kappa*I*(I>0)
        
        return ret
    
    def init_model(self):
        model=self.model
        
        #'closed-loop','urg-on-dec-mi','no-urg-mi','mi-urg','no-motor-fb'

        if model.find('mi')>=0:
            self.unc_add=-0.1
            
            for c in range(1,1+self.nchoice):
                for c2 in range(1,1+self.nchoice):
                    if (c==c2):
                        continue
                    self.set_edge(self.id_decision[c],self.id_decision[c2],0.0497,0)


        if model.find('urg-on-dec')>=0:
            
            for c in range(1,1+self.nchoice):
                for c2 in range(1,1+self.nchoice):
                    if (c==c2):
                        continue
                    self.set_edge(self.id_motor[c],self.id_decision[c2],0.0,0)
            self.get_urgency_signal=self.get_urgency_signal_on_dec
            self.COM_based_on=self.COM_based_on_dec
            self.get_decision=self.get_decision_by_dec
            self.thr = 20


        if model.find('no-urg')>=0:
            
            for c in range(1,1+self.nchoice):
                for c2 in range(1,1+self.nchoice):
                    if (c==c2):
                        continue
                    self.set_edge(self.id_motor[c],self.id_decision[c2],0.0,0)
            for c in range(1,1+self.nchoice):
                self.set_edge(self.id_unc[0],self.id_decision[c],0,1)
                
            self.get_urgency_signal=self.no_urgency_signal
            self.COM_based_on=self.COM_based_on_dec
            self.get_decision=self.get_decision_by_dec
            self.thr = 20

            


        if model.find('mi-with-ins-fb')>=0:
            
            # for c in range(1,1+self.nchoice):
            #     self.set_edge(self.id_unc[0],self.id_decision[c],0,1)
                
            self.get_urgency_signal=self.no_urgency_signal
            
        if model.find('mi-driven')>=0:
            
            for c in range(1,1+self.nchoice):
                self.set_edge(self.id_unc[0],self.id_decision[c],0,1)
                
            self.get_urgency_signal=self.no_urgency_signal
            
        if model.find('mi-urg')>=0:
            
            for c in range(1,1+self.nchoice):
                self.set_edge(self.id_unc[0],self.id_decision[c],0,1)
                
        if model.find('no-motor-fb')>=0:
            
            for c in range(1,1+self.nchoice):
                for c2 in range(1,1+self.nchoice):
                    if (c==c2):
                        continue
                    self.set_edge(self.id_motor[c],self.id_decision[c2],0.0,0)
            # self.exp_maxt=15000
            # self.maxt = int(self.exp_maxt/self.dt)+self.pret

        if model.find('no-dec-ff')>=0:
            
            for c in range(1,1+self.nchoice):
                for c2 in range(1,1+self.nchoice):
                    if (c==c2):
                        continue
                    self.set_edge(self.id_decision[c],self.id_motor[c2],0.0,0)
                    
        if model.find('no-ff-fb')>=0:
            
            for c in range(1,1+self.nchoice):
                for c2 in range(1,1+self.nchoice):
                    if (c==c2):
                        continue
                    self.set_edge(self.id_decision[c],self.id_motor[c2],0.0,0)
                    if (c==c2):
                        continue
                    self.set_edge(self.id_motor[c],self.id_decision[c2],0.0,0)
            # self.exp_maxt=15000
            # self.maxt = int(self.exp_maxt/self.dt)+self.pret
                    
            
    def init_setting(self):
        
        setting=self.setting
        # Sample rate
        if setting.find('hres')>=0:
            self.sample_rate=1/10
            self.nsamp=int(self.maxt*self.sample_rate)

        #SAT
        if setting.find('short')>=0:
            self.A_urg*=5
        if setting.find('long')>=0:
            self.A_urg=0
            self.exp_maxt=15000
            self.maxt = int(self.exp_maxt/self.dt)+self.pret
            self.nsamp=int(self.maxt*self.dt*self.sample_rate)
        if setting.find('medium')>=0:
            self.A_urg/=2
            self.exp_maxt=15000
            self.maxt = int(self.exp_maxt/self.dt)+self.pret
            self.nsamp=int(self.maxt*self.dt*self.sample_rate)

        # SAT by acc_rate
        if setting.find('slowacc')>=0:
            self.J_acc/=2
            self.exp_maxt=15000
            self.maxt = int(self.exp_maxt/self.dt)+self.pret
            self.nsamp=int(self.maxt*self.dt*self.sample_rate)
        if setting.find('fastacc')>=0:
            self.J_acc*=2


        

        import re
        s=re.split(' |_',setting)
        
        for i in range(len(s)):
            # Number of trials
            if s[i]=='repn':
                self.repn=int(s[i+1])

            # Stimulus baseline
            if s[i]=='input':
                self.input0*=float(s[i+1])

            # Max trial length
            if s[i]=='maxt':
                self.exp_maxt=float(s[i+1])
                self.maxt = int(self.exp_maxt/self.dt)+self.pret
                            
            # Stimulus offset delay
            if s[i]=='delay':
                self.delay=float(s[i+1])

            # Input noise scale
            if s[i]=='noise':
                self.input_noise=float(s[i+1])
           
            
