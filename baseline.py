

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
from tqdm import trange,tqdm

from experiment import Experiment
from analyzer import Analyzer

class BaselineExperiment(Experiment):
    """Class for running baseline experiments i.e. adding constant current input"""
    def __init__(self,**kwargs):
        super(BaselineExperiment, self).__init__(**kwargs)
        self.decision_add=0
        self.motor_add=0
        # self.unc_add=-0.2
        ori_unc_add=self.unc_add
        self.accunc_add=0

        if self.setting.find('decision')>=0:
            self.decision_add=-0.002
        if self.setting.find('motor')>=0:
            self.motor_add=-0.01
        if self.setting.find('unc')>=0:
            self.unc_add+=-0.015
            
        if self.setting.find('au')>=0:
            self.accunc_add=-10
            
        s=self.setting.split('_')
        for i in range(len(s)):
            if s[i]=='strength':
                scl=float(s[i+1])
                
                self.decision_add*=scl
                self.motor_add*=scl
                self.unc_add=ori_unc_add+scl*(self.unc_add-ori_unc_add)
                self.accunc_add*=scl

        self.mode='strong'
        
        if self.setting.find('weak')>=0:
            self.mode='weak'
        if self.setting.find('both')>=0:
            self.mode='both'

    def get_add(self,add,r):
        
        add[self.id_unc[0]] = self.unc_add
        add[self.id_accunc[0]] = self.accunc_add
        if self.mode=='both':
            add[self.id_decision[2]] = self.decision_add
            add[self.id_motor[2]] = self.motor_add
            add[self.id_decision[1]] = self.decision_add
            add[self.id_motor[1]] = self.motor_add
        elif self.mode=='weak':
            add[self.id_decision[1]] = self.decision_add
            add[self.id_motor[1]] = self.motor_add
        elif self.mode=='strong':
            add[self.id_decision[2]] = self.decision_add
            add[self.id_motor[2]] = self.motor_add
        else:
            print(f'Unknown mode {self.mode}, use mode strong instead')
            self.mode='strong'
            add[self.id_decision[2]] = self.decision_add
            add[self.id_motor[2]] = self.motor_add
        return add

        




if __name__=='__main__':

    from utils import run
    run(BaselineExperiment,Analyzer)

