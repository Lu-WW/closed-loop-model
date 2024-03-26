import numpy as np
class Alter():
    # functions for alternative models

    def get_urgency_signal_on_dec(self,r,I_net,t):
        I_net[self.id_decision[1]] += r[self.id_accunc[0],t-1]*self.A_urg
        I_net[self.id_decision[2]] += r[self.id_accunc[0],t-1]*self.A_urg
    def no_urgency_signal(self,r,I_net,t):
        pass


    def COM_based_on_dec(self):    
        return self.detailed_trial_data[:,:,self.id_decision[1]]-self.detailed_trial_data[:,:,self.id_decision[2]]

    def get_decision_by_dec(self,r,t):
        
        if (r[self.id_decision[1], t]) > self.thr:
            return 1
        elif (r[self.id_decision[2], t]) > self.thr:
            return 2
        else:
            return 0

    def local_urgency_signal(self,r,I_net,t):
        I_net[self.id_decision[1]] += r[self.id_unc[0],t-1]*self.A_urg
        I_net[self.id_decision[2]] += r[self.id_unc[0],t-1]*self.A_urg

    def get_DV_by_dec(self,r,t=None):
        scl=1
        if t:
            return np.mean(r[...,self.id_decision[2],t-1-self.wl_thr:t]-r[...,self.id_decision[1],t-1-self.wl_thr:t],-1)*scl
        else:
            return (r[...,self.id_decision[2],:]-r[...,self.id_decision[1],:])*scl



