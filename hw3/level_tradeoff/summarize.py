"""Produce plots and compact tables from frozen experiment outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SEEDS=(7,11,23)
ORDER=['A','B','C_p0','C_p0.25','C_p0.5','C_p0.75']
COLORS=['#555555','#17815c','#7255a4','#346fb0','#c77520','#b74166']
def rows(stage):
    data=json.loads((OUT/f'{stage}.json').read_text())
    return [(name,[data[f'{name}_s{s}'] for s in SEEDS]) for name in ORDER]
def vals(rs, key):
    if key=='flip': return np.array([np.mean([v['flip'] for v in r['gain'].values()]) for r in rs])*100
    if key=='HAP': return np.array([r['clean']['recall']['HAP'] for r in rs])*100
    return np.array([r['clean'][key] for r in rs])*100
def main():
    tables={}
    for stage in ('val','test'):
        rr=rows(stage);table=[]
        fig,axs=plt.subplots(1,2,figsize=(11,4.4))
        for ax,key,title in zip(axs,('accuracy','HAP'),('Overall accuracy','Happiness recall')):
            cx=[];cy=[]
            for (name,rs),color in zip(rr,COLORS):
                x,y=vals(rs,'flip'),vals(rs,key)
                ax.scatter(x,y,s=17,color=color,alpha=.35)
                ax.errorbar(x.mean(),y.mean(),xerr=x.std(),yerr=y.std(),fmt='o',color=color,capsize=3,label=name)
                if name.startswith('C'): cx.append(x.mean());cy.append(y.mean())
            ax.plot(cx,cy,color='#999999',lw=.8,zorder=0)
            ax.set_xscale('symlog',linthresh=.1)
            ax.set_xticks([0,.1,1,10,30],['0','0.1','1','10','30'])
            ax.set_xlim(-.03,40)
            ax.set(xlabel='Mean prediction flips under ±6 dB (%)',ylabel=title+' (%)',title=title)
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        axs[1].legend(fontsize=8,ncol=2)
        fig.suptitle(f'{stage.upper()} · level-channel dropout trade-off')
        fig.text(.5,.015,'Dots: seeds. Bars: seed SD. Lines: increasing p. X axis is linear near zero and logarithmic above 0.1%.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(OUT/f'{stage}-tradeoff.png',dpi=180);plt.close(fig)
        for name,rs in rr:
            r={'model':name,'accuracy':float(vals(rs,'accuracy').mean()),'accuracy_seed_sd':float(vals(rs,'accuracy').std()),
               'macro_f1':float(vals(rs,'macro_f1').mean()/100),'gain_flip':float(vals(rs,'flip').mean()),
               'gain_minus6_flip':float(np.mean([r['gain']['-6']['flip'] for r in rs])*100),
               'gain_plus6_flip':float(np.mean([r['gain']['6']['flip'] for r in rs])*100),
               'gain_probability_tv':float(np.mean([v['probability_tv'] for r in rs for v in r['gain'].values()]))}
            r.update({c:float(np.mean([r['clean']['recall'][c] for r in rs])*100) for c in ('ANG','DIS','FEA','HAP','NEU','SAD')})
            r['level_shift_flip']=None if name in ('A','B') else float(np.mean([v['flip'] for r in rs for v in r['level_only_intervention'].values()])*100)
            table.append(r)
        tables[stage]=table
        with open(OUT/f'{stage}-summary.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    (OUT/'summary.json').write_text(json.dumps(tables,indent=2))
    print(json.dumps(tables,indent=2))

def intervals(features):
    z=np.load(OUT/'test-predictions.npz');d=np.load(features)
    sp=d['speaker'][z['idx']];unique=np.unique(sp)
    boot=np.random.default_rng(37).integers(0,len(unique),(2000,len(unique)))
    audit={}
    selected=json.loads((OUT/'selection.json').read_text())['selected']['p']
    for b in ('A','B'):
        for c in (None,3):
            num=[];den=[]
            for actor in unique:
                m=sp==actor
                if c is not None: m=m&(z['y']==c)
                den.append(m.sum())
                num.append(np.mean([((z[f'C_p{selected:g}_s{s}'][m].argmax(1)==z['y'][m]).astype(float)
                                    -(z[f'{b}_s{s}'][m].argmax(1)==z['y'][m])).sum() for s in SEEDS]))
            num=np.array(num);den=np.array(den)
            delta=num[boot].sum(1)/den[boot].sum(1)*100
            audit[f'C_selected_minus_{b}_{"accuracy" if c is None else "HAP_recall"}']={
                'mean_difference_pp':float(num.sum()/den.sum()*100),
                'descriptive_actor_bootstrap_95_pp':np.quantile(delta,[.025,.975]).tolist()}
    (OUT/'paired-intervals.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--features',type=Path)
    args=parser.parse_args();OUT=args.out
    main()
    intervals(args.features or json.loads((OUT/'config.json').read_text())['features'])
