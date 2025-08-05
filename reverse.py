

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
from tqdm import trange,tqdm

from experiment import Experiment
from analyzer import Analyzer

class ReverseExperiment(Experiment):
    """Class for running reverse experiments i.e. stimuli show reversed coherence for a short period during a trial"""
    def __init__(self,**kwargs):
        super(ReverseExperiment, self).__init__(**kwargs)
 
        ## basic reverse parameters
        self.rev_start=200
        self.rev_len=160

        
    def get_input(self,r,I_net,t):
        

        ## apply revesed stimulus (coherence swapped, time flipped)
        if ((t-self.pret)*self.dt>self.rev_start and (t-self.pret)*self.dt<=self.rev_start+self.rev_len):
            rev_t=2*int(self.rev_start/self.dt+self.pret)-t
            r[self.id_sti[2],t],r[self.id_sti[1],t]=r[self.id_sti[1],rev_t],r[self.id_sti[2],rev_t]
        
        I_net[self.id_decision[1]] += r[self.id_sti[1],t]
        I_net[self.id_decision[2]] += r[self.id_sti[2],t]


if __name__=='__main__':

    from utils import run
    run(ReverseExperiment,Analyzer)
