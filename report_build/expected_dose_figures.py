from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / 'reports' / 'expected_dose_figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':True,'axes.spines.right':True,'axes.titleweight':'bold','savefig.facecolor':'white'})
rng = np.random.default_rng(20261105)
groups=['No HA','1 nM HA','5 nM HA','10 nM HA']
colors=['#606060','#4289B0','#C444AC','#22AEBB']
subtitle='Dose comparison (B) | 1383D2 | Seeding density: 2.5 × 10³ cells/cm²'
schedule='HA groups: once daily from Day −3 to Day 0'

def save(fig, name):
    fig.savefig(OUT/(name+'.png'),dpi=220)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)

fig,ax=plt.subplots(figsize=(8.6,6.4))
fig.subplots_adjust(top=.78,bottom=.24,left=.13,right=.96)
fig.suptitle('Nuclear phenotype composition',y=.97,fontsize=17,fontweight='bold')
fig.text(.5,.915,subtitle,ha='center',fontsize=10)
fig.text(.5,.877,schedule,ha='center',fontsize=10,color='#555555')
fractions=np.array([[.34,.49,.17],[.40,.43,.17],[.52,.31,.17],[.62,.23,.15]])
bottom=np.zeros(4)
for i,(label,c) in enumerate(zip(['Phenotype A','Phenotype B','Phenotype C'],['#087CB5','#E5A000','#079B78'])):
    ax.bar(np.arange(4),fractions[:,i],bottom=bottom,color=c,width=.62,label=label)
    bottom+=fractions[:,i]
ax.set(xticks=np.arange(4),xticklabels=groups,ylim=(0,1.05),ylabel='Cell fraction',xlabel='HA condition')
ax.set_yticks(np.arange(0,1.01,.2))
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False)
fig.text(.5,.055,'SIMULATED EXPECTED PATTERN — not experimental results',ha='center',fontsize=10,fontweight='bold',color='#8C3B16')
fig.text(.5,.02,'Phenotype A–C are illustrative clusters; no biological identity or sample size is implied.',ha='center',fontsize=9,color='#555555')
save(fig,'01_phenotype_composition')

features=['Neighbor heterogeneity','Local crowding','Nuclear spacing','Circularity','Elongation','Nuclear area']
effects=np.array([[.04,.10,-.06,-.05,.01,-.09],[.10,.23,-.17,-.15,.04,-.25],[.16,.34,-.25,-.22,.06,-.35]])
fig,axs=plt.subplots(1,3,figsize=(12.4,6.3),sharey=True,sharex=True)
fig.subplots_adjust(top=.77,bottom=.27,left=.20,right=.98,wspace=.14)
fig.suptitle('Nuclear and spatial feature shifts relative to No HA',y=.97,fontsize=17,fontweight='bold')
fig.text(.5,.915,subtitle,ha='center',fontsize=10)
fig.text(.5,.877,schedule,ha='center',fontsize=10,color='#555555')
for i,ax in enumerate(axs):
    c=colors[i+1]
    for j,mu in enumerate(effects[i]):
        offsets=np.array([-.07,.00,.07]) * (1+.2*((i+j)%3))
        vals=mu+offsets
        ax.plot([vals.min(),vals.max()],[j,j],color=c,alpha=.4,lw=2)
        ax.scatter(vals,np.array([j-.10,j,j+.10]),color=c,s=22,alpha=.55,zorder=3)
        ax.scatter(mu,j,marker='D',color=c,s=60,zorder=4)
        ax.annotate(f'{mu:+.2f}',(mu,j),xytext=(7,-14),textcoords='offset points',color=c,fontsize=9,fontweight='bold')
    ax.axvline(0,color='#777777',ls=':',lw=1.2)
    ax.grid(axis='x',alpha=.15)
    ax.set(title=groups[i+1],xlim=(-.65,.65),xticks=[-.6,-.3,0,.3,.6],yticks=np.arange(6),yticklabels=features)
axs[0].invert_yaxis()
fig.text(.59,.17,'Feature shift relative to No HA (No HA IQR units)',ha='center',fontsize=11)
fig.text(.5,.095,'Diamonds: illustrative medians | Dots: 3 simulated well summaries | Lines: simulated ranges',ha='center',fontsize=9,color='#555555')
fig.text(.5,.05,'SIMULATED EXPECTED PATTERN — not experimental results',ha='center',fontsize=10,fontweight='bold',color='#8C3B16')
fig.text(.5,.015,'Effect directions and magnitudes are hypothetical; lines are not confidence intervals.',ha='center',fontsize=9,color='#555555')
save(fig,'02_feature_shifts')

fig,ax=plt.subplots(figsize=(8.6,6.4))
fig.subplots_adjust(top=.78,bottom=.23,left=.15,right=.96)
fig.suptitle('YAP nuclear-to-cytoplasmic distribution',y=.97,fontsize=17,fontweight='bold')
fig.text(.5,.915,subtitle,ha='center',fontsize=10)
fig.text(.5,.877,schedule,ha='center',fontsize=10,color='#555555')
data=[rng.normal(mu,sd,240) for mu,sd in zip([.58,.45,.27,.12],[.31,.34,.37,.40])]
bp=ax.boxplot(data,positions=np.arange(4),widths=.5,patch_artist=True,showfliers=False,medianprops={'color':'#F2B45A','linewidth':2},whiskerprops={'color':'#555555'},capprops={'color':'#555555'})
for patch,c in zip(bp['boxes'],colors): patch.set(facecolor=c,edgecolor='#444444',alpha=.88)
ax.axhline(0,color='#777777',ls=':',lw=1.2)
ax.set(xticks=np.arange(4),xticklabels=groups,ylabel='log₂(YAP nuclear / cytoplasmic intensity)',xlabel='HA condition',ylim=(-1.2,1.6))
fig.text(.5,.105,'Boxes: median and IQR | Whiskers: 1.5 × IQR | Outliers omitted',ha='center',fontsize=9,color='#555555')
fig.text(.5,.06,'SIMULATED EXPECTED PATTERN — not experimental results',ha='center',fontsize=10,fontweight='bold',color='#8C3B16')
fig.text(.5,.022,'Synthetic cell distributions do not establish treatment effects or independent biological replication.',ha='center',fontsize=9,color='#555555')
save(fig,'03_yap_distribution')
print(OUT)
