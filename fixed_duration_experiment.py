from experiment import *

class FixedDurationExperiment(Experiment):
  
    """ Class for running fixed duration experiments"""
    def __init__(self,**kwargs):
        super(FixedDurationExperiment, self).__init__(**kwargs)

        # infinite thr to avoid early decision and trial end
        self.thr=1e9

        # No urgency signal applied
        self.A_urg=0



    def analyze_trial(self,r,t):
        data=np.zeros(self.trial_data.shape[-1])
        wt=int(10/self.dt)   
        t=r.shape[-1]-1
        delt=np.mean(r[self.id_motor[1], t-wt:t+1] - r[self.id_motor[2],  t-wt:t+1])
        if delt>0:
            res=1
        else:
            res=2
            
        data[0] = res
        data[1] = t
        data[2] = np.mean(r[self.id_decision[res], t-wt:t+1] - r[self.id_decision[3-res],  t-wt:t+1])
        data[3] = np.mean(r[self.id_accunc[0], t-wt:t+1])
        data[4] = np.mean(r[self.id_motor[res], t-wt:t+1] - r[self.id_motor[3-res],  t-wt:t+1])
        data[5] = np.mean(r[self.id_unc[0], t-wt:t+1])
        return data
    
if __name__=='__main__':

    from utils import run
    run(FixedDurationExperiment,Analyzer)
