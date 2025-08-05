
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
from tqdm import trange,tqdm

from base import Base
from analyzer import Analyzer
from alter import Alter
class Experiment(Base,Alter):
    """Basic class for running experiments"""
    def __init__(self,**kwargs):
        super(Experiment, self).__init__(**kwargs)

        ## for parallelization
        self.n_pool=16
    def init_sti(self,input_coh,noise,r):
        ## initialize stimuli
        input1 = self.input0*(1-input_coh)
        input2 = self.input0*(1+input_coh)

        for t in range(self.pret, self.maxt):
            r[self.id_sti[1],t]=input1+self.input_noise*noise[self.id_sti[1],t-1]+self.input_baseline
            r[self.id_sti[2],t]=input2+self.input_noise*noise[self.id_sti[2],t-1]+self.input_baseline

    def get_input(self,r,I_net,t):
        ## apply stimulus input to decision modules
        I_net[self.id_decision[1]] += r[self.id_sti[1],t]
        I_net[self.id_decision[2]] += r[self.id_sti[2],t]

    def get_urgency_signal(self,r,I_net,t):
        ## apply urgency signal to motor modules
        I_net[self.id_motor[1]] += r[self.id_accunc[0],t-1]*self.A_urg
        I_net[self.id_motor[2]] += r[self.id_accunc[0],t-1]*self.A_urg

    def get_decision(self,r,t):
        ## get model choice based on motor modules
        
        if (r[self.id_motor[1], t]) > self.thr:
            return 1
        elif (r[self.id_motor[2], t]) > self.thr:
            return 2
        else:
            return 0
    
    def get_add(self,add,r):
        ## get altered baseline for instantaneous uncertainty
        add[self.id_unc[0]] = self.unc_add
        return add

    def run_one(self,input_coh):
        ## run a trial
        I = np.zeros((self.n_dyn, self.maxt))
        S = np.zeros((self.n_dyn, self.maxt))
        r = np.zeros((self.n_pop, self.maxt))
        noise = np.zeros((self.n_pop, self.maxt))
        for t in range(1, self.maxt):
            noise[:, t] = noise[:, t - 1] + (-noise[:, t - 1] + np.sqrt(self.tau_noise/self.dt) * self.sigma_noise * np.random.randn(noise.shape[0])) / self.tau_noise * self.dt


        add=np.zeros(self.n_dyn)
        add=self.get_add(add,r)
        self.init_sti(input_coh,noise,r)
        rt=-1
        decision_made=False
        for t in range(1, self.maxt):
            ## update for each time step
            S[:, t] = S[:, t - 1] + (-S[:, t - 1] / self.tau_N + self.gamma *(1 - S[:, t - 1]) * r[:self.n_dyn,t - 1] / 1000) * self.dt
            I_net = S[:, t - 1] @ self.W 

            I_net+=add

            if t > self.pret and t < self.maxt:
                if not decision_made:
                    self.get_input(r,I_net,t)

                    
            self.get_urgency_signal(r,I_net,t)
                
                

            I[:, t] = I[:,t] + self.I_0 + I_net +noise[:self.n_dyn,t]
            
            r[:self.n_dyn, t] = r[:self.n_dyn,t - 1] + (-r[:self.n_dyn,t - 1] + self.Phy(I[:, t])) / self.tau_r * self.dt           
            r[self.id_accunc[0],t]=r[self.id_accunc[0],t-1]+(-r[self.id_accunc[0],t-1]+self.Phy_acc((self.J_acc*S[self.id_unc[0],t-1]+I_net[self.id_accunc[0]])))/self.tau_acc*self.dt
            
            
            if not decision_made and t>=int(self.delay/self.dt) and self.get_decision(r,t-int(self.delay/self.dt)):
                # check choice
                decision_made=True
                rt=t-int(self.delay/self.dt)
                r[self.id_sti[1],t+1:]=0
                r[self.id_sti[2],t+1:]=0
                
        return r,rt

    def analyze_trial(self,r,t):
        ## prepare trial behavioral results
        data=np.zeros(self.trial_data.shape[-1])
        wt=int(10/self.dt)   
        if t<0:

            t=r.shape[-1]
            
            data[0] = 0
            data[1] = -1000
            data[2] = np.abs(np.mean(r[self.id_decision[2], t-wt:t+1] - r[self.id_decision[1],  t-wt:t+1]))
            data[3] = np.mean(r[self.id_accunc[0], t-wt:t+1])
            data[4] = np.abs(np.mean(r[self.id_motor[2], t-wt:t+1] - r[self.id_motor[1],  t-wt:t+1]))
            data[5] = np.mean(r[self.id_unc[0], t-wt:t+1])
            return data
        
        res=self.get_decision(r,t)
        
        data[0] = res
        data[1] = t
        data[2] = np.mean(r[self.id_decision[res], t-wt:t+1] - r[self.id_decision[3-res],  t-wt:t+1])
        data[3] = np.mean(r[self.id_accunc[0], t-wt:t+1])
        data[4] = np.mean(r[self.id_motor[res], t-wt:t+1] - r[self.id_motor[3-res],  t-wt:t+1])
        data[5] = np.mean(r[self.id_unc[0], t-wt:t+1])
        
        return data

    def getdata(self,coh,rep,r,t,detailed=True):
        ## prepare data to save baeed on parallelization

        shm_trial_data=shared_memory.SharedMemory(name=self.shm_names[0])
        trial_data=np.ndarray(self.trial_data.shape,buffer=shm_trial_data.buf)
        trial_data[coh,rep]=self.analyze_trial(r,t)
        if not detailed:
            return
            
        shm_detailed_trial_data=shared_memory.SharedMemory(name=self.shm_names[1])
        detailed_trial_data=np.ndarray(self.detailed_trial_data.shape,buffer=shm_detailed_trial_data.buf)
        from utils import down_sample
        detailed_trial_data[coh,rep]=down_sample(r,self.nsamp)
        self.get_other_info(detailed_trial_data,r,coh,rep)
        

                
    def run_batch(self,coh,rep_l,rep_r,input_coh,detailed=True):
        np.random.seed()
        for rep in range(rep_l,rep_r):
            r,t=self.run_one(input_coh)
            self.getdata(coh,rep,r,t,detailed)
            


    
    def run_experiment(self,detailed=True):      
        ## run with parrallelization
        shm_trial_data = shared_memory.SharedMemory(create=True, size=self.trial_data.nbytes)
        self.trial_data = np.ndarray(self.trial_data.shape, buffer=shm_trial_data.buf)
        self.trial_data[:] = 0
        self.shm_names=[shm_trial_data.name]
        
        if detailed:
            self.detailed_trial_data=np.zeros((self.coh_level,self.repn,self.n_pop, self.nsamp))
            shm_detailed_trial_data = shared_memory.SharedMemory(create=True, size=self.detailed_trial_data.nbytes)
            self.detailed_trial_data = np.ndarray(self.detailed_trial_data.shape, buffer=shm_detailed_trial_data.buf)
            self.detailed_trial_data[:] = 0
            self.shm_names.append(shm_detailed_trial_data.name)

        for coh,input_coh in enumerate(tqdm(self.coh_list)):

            p=Pool(self.n_pool)           

            results=[p.apply_async(self.run_batch,args=(coh,int(i*self.repn/self.n_pool),int((i+1)*self.repn/self.n_pool),input_coh,detailed),error_callback=print) for i in range(self.n_pool)]
            
            p.close()
            p.join()
         

        self.save_data(detailed=detailed)
            

    def save_data(self,detailed=True):
        np.save(f'{self.trial_data_name}',self.trial_data)
        shm_trial_data=shared_memory.SharedMemory(name=self.shm_names[0])
        shm_trial_data.close()
        shm_trial_data.unlink()

        if detailed:
            np.save(f'{self.detailed_trial_data_name}',self.detailed_trial_data) 
            shm_detailed_trial_data=shared_memory.SharedMemory(name=self.shm_names[1])
            shm_detailed_trial_data.close()
            shm_detailed_trial_data.unlink()
            

    def get_other_info(self,detailed_trial_data,r,coh,rep):
        pass




if __name__=='__main__':

    from utils import run
    run(Experiment,Analyzer)
