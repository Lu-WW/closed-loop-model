import os
import copy


ori_path=os.getcwd()

basic_setting_list=['normal','noise_0.5','noise_1','noise_1.5','noise_2','noise_3']
basic_setting_list+=['long','short','short_delay_200','delay_200']

basic_command_list=[f'experiment.py -s {s}'for s in basic_setting_list]

basic_setting_list+=['reverse','reverse noise_1','reverse noise_2','reverse noise_3','pulse para']
basic_command_list+=['reverse.py','reverse.py -s noise_1','reverse.py -s noise_2','reverse.py -s noise_3','pulse_para.py']



model_list=['closed-loop','urg-on-dec-mi','no-urg-mi']
model_list+=['no-motor-fb','no-dec-ff','no-ff-fb']
model_list+=['mi-driven','mi-urg']
model_list+=['motor-on-meta']


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("-m","--model",default=None)
parser.add_argument("-e","--eval",action='store_true')

args=parser.parse_args()
if args.model:
    model_list=[args.model]
    
for model in model_list:
    setting_list=copy.deepcopy(basic_setting_list)
    command_list=copy.deepcopy(basic_command_list)
    if model=='closed-loop':
        setting_list+=['decision baseline','motor baseline','unc baseline','accunc baseline']
        command_list+=['baseline.py -s decision','baseline.py -s motor','baseline.py -s unc','baseline.py -s au']

        setting_list+=['attention']
        command_list+=['attention.py']


        setting_list+=['motor baseline weak',]
        command_list+=['baseline.py -s motor_weak',]


        # setting_list+=['fixed_duration noise_1','fixed_duration noise_2','fixed_duration noise_3']
        # command_list+=['fixed_duration_experiment.py -s maxt_500_noise_1',
        #                'fixed_duration_experiment.py -s maxt_500_noise_2',
        #                'fixed_duration_experiment.py -s maxt_500_noise_3']
        # setting_list+=['fixed_duration maxt_500','fixed_duration maxt_1000','fixed_duration maxt_3000']
        # command_list+=['fixed_duration_experiment maxt_500_noise_1',
        #                'fixed_duration_experiment maxt_1000_noise_1',
        #                'fixed_duration_experiment maxt_3000_noise_1']


        # setting_list+=['accunc baseline strength -1']
        # command_list+=['baseline.py -s au_strength_-1']

        # setting_list+=['unc baseline strength -1']
        # command_list+=['baseline.py -s unc_strength_-1']

        # setting_list+=['decision baseline weak','decision baseline both','motor baseline both']
        # command_list+=['baseline.py -s decision_weak','baseline.py -s decision_both','baseline.py -s motor_both']

        # setting_list+=['attention dec','attention mot']
        # command_list+=['attention.py -s dec','attention.py -s mot']

        # setting_list+=['slowacc','fastacc']
        # command_list+=['experiment.py -s slowacc','experiment.py -s fastacc']



    # if model=='no-motor-fb' or model=='no-dec-ff' or model=='no-ff-fb' or model=='motor-on-meta':

    #     setting_list+=['motor baseline']
    #     command_list+=['baseline.py -s motor']
    #     setting_list+=['motor baseline weak','motor baseline both']
    #     command_list+=['baseline.py -s motor_weak','baseline.py -s motor_both']

    print(f'{model} model')

    if not os.path.isdir(f'./{model}'):
        os.mkdir(f'./{model}')

    for i,setting in enumerate(setting_list):
        command=command_list[i]
        print(f'run {setting}')
        if args.eval:
            os.system(f'python {ori_path}/{command} -m {model} -e >>"./{model}/log.out"')
        else:
            os.system(f'python {ori_path}/{command} -m {model} >>"./{model}/log.out"')
    
    os.system(f'python {ori_path}/compare.py -m {model} >>"./{model}/log.out"')

    os.system(f'python {ori_path}/GLM.py -m {model} >>"./{model}/log.out"')
