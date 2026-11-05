from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'reports'/'expected_AB_figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,'axes.titleweight':'bold','savefig.facecolor':'white'})
rng=np.random.default_rng(20261105)
features=['Neighbor heterogeneity','Local crowding','Nuclear spacing','Circularity','Elongation','Nuclear area']
specs={
 'A':dict(title='Mechanism experiment',groups=['No HA','Single exposure','Daily exposure'],
          fractions=[[.34,.49,.17],[.45,.38,.17],[.61,.24,.15]],
          effects=[[.06,.14,-.10,-.08,.02,-.15],[.15,.32,-.24,-.21,.05,-.33]],
          means=[.58,.38,.14],sds=[.31,.35,.40],colors=['#606060','#C444AC','#22AEBB']),
 'B':dict(title='Dose comparison',groups=['No HA','1 nM HA','5 nM HA','10 nM HA'],
          fractions=[[.34,.49,.17],[.40,.43,.17],[.52,.31,.17],[.62,.23,.15]],
          effects=[[.04,.10,-.06,-.05,.01,-.09],[.10,.23,-.17,-.15,.04,-.25],[.16,.34,-.25,-.22,.06,-.35]],
          means=[.58,.45,.27,.12],sds=[.31,.34,.37,.40],colors=['#606060','#4289B0','#C444AC','#22AEBB'])
}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.14)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.14)
    plt.close(fig)
for key,s in specs.items():
    x=np.arange(len(s['groups']))
    xlabel='HA exposure schedule' if key=='A' else 'HA condition'
    fig,ax=plt.subplots(figsize=(7.2,4.7),layout='constrained')
    bottom=np.zeros(len(x))
    vals=np.array(s['fractions'])
    for j,c in enumerate(['#087CB5','#E5A000','#079B78']):
        ax.bar(x,vals[:,j],bottom=bottom,width=.60,color=c)
        bottom+=vals[:,j]
    ax.set(title=f'{key} | Nuclear phenotype composition',xticks=x,xticklabels=s['groups'],ylim=(0,1.05),yticks=np.arange(0,1.01,.2),ylabel='Cell fraction',xlabel=xlabel)
    save(fig,f'{key}1_phenotype_composition')

    k=len(s['effects'])
    fig,axs=plt.subplots(1,k,figsize=(3.4*k+2.1,4.8),sharey=True,sharex=True)
    fig.subplots_adjust(left=.24 if k==2 else .19,right=.98,bottom=.18,top=.78,wspace=.13)
    fig.suptitle(f'{key} | Nuclear and spatial feature shifts',fontsize=16,fontweight='bold',y=.98)
    for i,ax in enumerate(axs):
        for j,mu in enumerate(s['effects'][i]):
            vals=mu+np.array([-.07,0,.07])*(1+.2*((i+j)%3))
            ax.plot([vals.min(),vals.max()],[j,j],color=s['colors'][i+1],alpha=.45,lw=2)
            ax.scatter(vals,[j-.08,j,j+.08],s=22,color=s['colors'][i+1],alpha=.55,zorder=3)
            ax.scatter(mu,j,marker='D',s=60,color=s['colors'][i+1],zorder=4)
        ax.axvline(0,color='#777777',ls=':',lw=1.2)
        ax.grid(axis='x',alpha=.14)
        ax.set(title=s['groups'][i+1],xlim=(-.6,.6),xticks=[-.6,-.3,0,.3,.6],yticks=np.arange(6),yticklabels=features,ylim=(5.5,-.5))
    fig.text(.6 if k==2 else .58,.035,'Change relative to No HA (IQR units)',ha='center',fontsize=12)
    save(fig,f'{key}2_feature_shifts')

    fig,ax=plt.subplots(figsize=(7.2,4.7),layout='constrained')
    data=[rng.normal(mu,sd,240) for mu,sd in zip(s['means'],s['sds'])]
    bp=ax.boxplot(data,positions=x,widths=.5,patch_artist=True,showfliers=False,medianprops={'color':'#F2B45A','linewidth':2},whiskerprops={'color':'#555555'},capprops={'color':'#555555'})
    for patch,c in zip(bp['boxes'],s['colors']):patch.set(facecolor=c,edgecolor='#444444',alpha=.88)
    ax.axhline(0,color='#777777',ls=':',lw=1.2)
    ax.set(title=f'{key} | YAP nuclear-to-cytoplasmic ratio',xticks=x,xticklabels=s['groups'],ylabel='log₂(YAP nuclear / cytoplasmic intensity)',xlabel=xlabel,ylim=(-1.2,1.6))
    save(fig,f'{key}3_yap_distribution')
print(OUT)
