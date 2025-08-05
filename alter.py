import numpy as np
class Alter():
    # functions for alternative models

    def get_urgency_signal_on_dec(self,r,I_net,t):
        ## apply urgency signal from accumulated uncertainty to decision modules
        I_net[self.id_decision[1]] += r[self.id_accunc[0],t-1]*self.A_urg
        I_net[self.id_decision[2]] += r[self.id_accunc[0],t-1]*self.A_urg
    def no_urgency_signal(self,r,I_net,t):
        pass


    def COM_based_on_dec(self):    
        ## calculate the changes of mind based on decision modules
        return self.detailed_trial_data[:,:,self.id_decision[1]]-self.detailed_trial_data[:,:,self.id_decision[2]]

    def get_decision_by_dec(self,r,t):
        ## get model choice based on decision modules
        
        if (r[self.id_decision[1], t]) > self.thr:
            return 1
        elif (r[self.id_decision[2], t]) > self.thr:
            return 2
        else:
            return 0

    def local_urgency_signal(self,r,I_net,t):
        ## apply urgency signal from instantaneous uncertainty to decision modules

        I_net[self.id_decision[1]] += r[self.id_unc[0],t-1]*self.A_urg
        I_net[self.id_decision[2]] += r[self.id_unc[0],t-1]*self.A_urg

    def get_DV_by_dec(self,r,t=None):
        ## get decision variable based on decision modules
        scl=1
        if t:
            return np.mean(r[...,self.id_decision[2],t-1-self.wl_thr:t]-r[...,self.id_decision[1],t-1-self.wl_thr:t],-1)*scl
        else:
            return (r[...,self.id_decision[2],:]-r[...,self.id_decision[1],:])*scl


    def run_one_with_motor_on_meta(self,input_coh):
        ## added motor to accumulated uncertainty effect
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
            
            S[:, t] = S[:, t - 1] + (-S[:, t - 1] / self.tau_N + self.gamma *(1 - S[:, t - 1]) * r[:self.n_dyn,t - 1] / 1000) * self.dt
            I_net = S[:, t - 1] @ self.W 

            I_net+=add

            if t > self.pret and t < self.maxt:
                if not decision_made:
                    self.get_input(r,I_net,t)

                    
            self.get_urgency_signal(r,I_net,t)
                
                

            I[:, t] = I[:,t] + self.I_0 + I_net +noise[:self.n_dyn,t]
            
            r[:self.n_dyn, t] = r[:self.n_dyn,t - 1] + (-r[:self.n_dyn,t - 1] + self.Phy(I[:, t])) / self.tau_r * self.dt           
            r[self.id_accunc[0],t]=r[self.id_accunc[0],t-1]+(-r[self.id_accunc[0],t-1]+self.Phy_acc((
                self.J_acc*S[self.id_unc[0],t-1]+self.J_m2acc*S[self.id_motor[1],t-1]+self.J_m2acc*S[self.id_motor[2],t-1]+I_net[self.id_accunc[0]])))/self.tau_acc*self.dt
            
            
            if not decision_made and t>=int(self.delay/self.dt) and self.get_decision(r,t-int(self.delay/self.dt)):
                decision_made=True
                rt=t-int(self.delay/self.dt)
                r[self.id_sti[1],t+1:]=0
                r[self.id_sti[2],t+1:]=0
                
        return r,rt


