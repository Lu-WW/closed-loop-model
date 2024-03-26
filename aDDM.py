import numpy as np
import matplotlib.pyplot as plt


def run_aDDM(coh,att_list,dist,rt,input0=1,unatt_coef=0.7,noise_sigma=1,bound=100,maxt=100):
  input1 = input0*(1-coh)
  input2 = input0*(1+coh)
  k=100
  data=np.zeros(rt*k)
  input=np.zeros(rt*k)
  att=0
  i=0
  
  raw_list_len=len(att_list)
  
  for t in range(1,rt*k):




    if i<len(att_list) and t*a.dt>=att_list[i][0]:
        att=att_list[i][1]
        i+=1

    
    if i>=len(att_list):
      nxt_att=3-att
      l=np.random.choice(dist)
      att_list.append((t+l,nxt_att))

    if att==0:
      input[t]=0
    if att==1:
      input[t]=input2*unatt_coef-input1
    if att==2:
      input[t]=input2-input1*unatt_coef

    data[t]=data[t-1] + input[t] + noise_sigma*np.random.randn()


    if np.abs(data[t])>bound:
      data=data[:t+1]
      input=input[:t+1]
      att_list.pop()
      break

  return data,input,t,att_list

def get_data():
  
  choice=np.zeros_like(a.choice)
  model_rt=np.zeros_like(a.choice)
  
  from tqdm import trange
  all_att_list=[]
  for c in trange(a.coh_level):
    coh=a.coh_list[c]
    
    all_att_list.append([])
    for r in trange(a.repn):
      l=a.get_att_list(c,r)
      rt=int(a.raw_rt[c,r]-a.pret)
      if rt==0:
        all_att_list[c].append(l)
        continue
      data,input,t,l=run_aDDM(coh,l,mid_len[c],rt)
      choice[c,r]=int(data[-1]>0)+1
      model_rt[c,r]=int(t+1+a.pret)
      all_att_list[c].append(l)
    print(f'Coherence {coh} accuracy {np.mean(choice[c])}')
  np.save('aDDM_choice.npy',choice)
  np.save('aDDM_rt.npy',model_rt)
  np.save('aDDM_att_list.npy',all_att_list)

def draw_aDDM_example(c,r):
  coh=a.coh_list[c]
  l=a.get_att_list(c,r)
  rt=int(a.raw_rt[c,r]-a.pret)
  np.random.seed(100)
  data,input,rt,l=run_aDDM(coh,l,mid_len[c],rt)
  plt.figure(figsize=(10,5))
  axes=plt.axes()
  # axes.get_xaxis().set_visible(False)
  # axes.get_yaxis().set_visible(False)
  axes.spines['bottom'].set_visible(False)
  plt.xticks([])
  plt.yticks([])
  plt.plot(data,marker=',')
  m=np.max(np.abs(data))
  text_y=105
  def draw_area(l,r,att):
    if att==1:
      plt.fill_between([l,r],[-100,-100],[100,100],alpha=0.2,color='lightblue',lw=0)
      plt.text((l+r)/2,text_y,'Left',ha='center')
    if att==2:
      plt.fill_between([l,r],[-100,-100],[100,100],alpha=0.2,color='steelblue',lw=0)
      plt.text((l+r)/2,text_y,'Right',ha='center')
  for i in range(len(l)-1):
    draw_area(l[i][0],l[i+1][0],l[i][1])
  draw_area(l[-1][0],rt,l[-1][1])
  plt.axhline(0,linestyle='dashed',color='black',marker=',',linewidth=2)
  plt.scatter(rt,data[-1],color='purple',zorder=3,label='Decision')
  plt.ylim(-m*1.1,m*1.1)
  
  plt.legend(loc=(0.7,1.1))
  plt.xlabel('Time',labelpad=20)
  plt.ylabel('Decision value')
  plt.savefig('aDDM_example')
  plt.close()
  
if __name__=='__main__':
  model='closed-loop'
  setting='attention'
  import os
  os.chdir(f'./{model}/{setting}')
  from attention import AttentionAnalyzer
  a= AttentionAnalyzer()
  a.prepare()
  first_len,mid_len,last_len=a.attention_analysis()
  os.chdir('../..')

  if not os.path.isdir(f'./example/aDDM'):
    os.makedirs(f'./example/aDDM')
  os.chdir('./example/aDDM')

  draw_aDDM_example(2,100)
  get_data()


  choice=np.load('aDDM_choice.npy')
  rt=np.load('aDDM_rt.npy')
  att_list=np.load('aDDM_att_list.npy',allow_pickle=True)
  f=np.zeros((2,2))
  for i in range(2):
    for j in range(2):
      f[i][j]=np.sum((choice==i+1)*(a.choice==j+1))
  f/=np.sum(f,1,keepdims=True)
  fig=plt.figure()
  plt.imshow(f,cmap='YlGnBu',aspect='equal',vmin=0,vmax=1)
  cb=plt.colorbar()
  for i in range(2):
    for j in range(2):
      c='black'
      if f[i][j]>0.5:
        c='white'
      plt.text(i,j,"%.2f"%f[i][j],ha='center',va='center',color=c)

  plt.xticks([0,1],[1,2])
  plt.yticks([0,1],[1,2])
  plt.xlabel('Model choice')
  plt.ylabel('aDDM choice')
  cb.set_label('Frequency')
  fig.savefig('choice_comparision')
  plt.close(fig)
  a.choice=choice
  a.raw_rt=rt
  
  a.reaction_time=(a.raw_rt-a.pret)*a.dt
  a.miss=a.choice==0
  a.attention_analysis(att_list=att_list,legend=False)
  

