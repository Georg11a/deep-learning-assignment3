"""HW3 level-channel dropout study. See protocol-zh.txt before running."""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF
from sklearn.metrics import confusion_matrix, f1_score
from baseline_model import SegmentMLP
import features as F

ROOT = Path(__file__).resolve().parent
SEEDS = (7, 11, 23)
PS = (0., .25, .5, .75)
LEVEL = [F.FEATURE_NAMES.index(c) for c in ('mfcc0_mean', 'logrms_mean')]

def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def split(speakers):
    rest = [s for s in sorted(set(speakers)) if s not in ('1001', '1002')]
    np.random.default_rng(7).shuffle(rest)
    groups = {'test': ['1001','1002']+rest[:16], 'val': rest[16:25], 'train': rest[25:]}
    return {k: np.flatnonzero(np.isin(speakers, v)) for k,v in groups.items()}, groups

class LevelModel(SegmentMLP):
    def __init__(self, p=0.):
        super().__init__(88, dropout=.5)
        self.p = p
        self.level_net = nn.Sequential(nn.Linear(2,16,bias=False),nn.ReLU(),nn.Linear(16,6,bias=False))
        self.gate = nn.Linear(64,1)
        self.mask_rng = torch.Generator()

    def components(self, x):
        mu = x[:,:,LEVEL].mean(1)
        rel = x.clone()
        rel[:,:,LEVEL] -= mu[:,None,:]
        z = self.encoder(rel)
        return self.classify(z), torch.sigmoid(self.gate(z.mean(1))), mu

    def forward(self, x, strength=1., mu_shift=None):
        relative, gate, mu = self.components(x)
        if mu_shift is not None:
            mu = mu + mu_shift
        contribution = gate*self.level_net(mu)
        if self.training:
            mask = (torch.rand((len(x),1),generator=self.mask_rng) >= self.p).to(x.dtype)
            contribution = contribution*mask
        return relative + strength*contribution

def wave_work(path):
    """Matched clean and global gain features; F0 is held fixed as in the pilot."""
    y,_ = F.trim(F.load(path))
    ff0 = F.frame_features(y)
    out = []
    for db in (0,-6,6):
        if db == 0:
            ff = ff0
        else:
            yy = y*10**(db/20)
            # Extract without rerunning YIN, whose clean track is deliberately held fixed.
            import librosa
            m = librosa.feature.mfcc(y=yy,sr=F.SR,n_mfcc=20,n_fft=512,hop_length=160)
            width = min(9,m.shape[1]-(1-m.shape[1]%2)) if m.shape[1]>=3 else 0
            dm = librosa.feature.delta(m,width=width) if width>=3 else np.zeros_like(m)
            rms = librosa.feature.rms(y=yy,frame_length=512,hop_length=160)[0]
            n = len(ff0['logrms']); lr = np.log(rms[:n]+1e-5)
            f0 = np.exp(ff0['logf0'])
            ff = {'mfcc':m[:,:n].T,'dmfcc':dm[:,:n].T,'logrms':lr,'logf0':ff0['logf0'],
                  'voiced':(lr>np.percentile(lr,30))&(f0>F.FMIN*1.05)&(f0<F.FMAX*.95),
                  'cent':librosa.feature.spectral_centroid(y=yy,sr=F.SR,n_fft=512,hop_length=160)[0][:n]/(F.SR/2),
                  'zcr':librosa.feature.zero_crossing_rate(yy,frame_length=512,hop_length=160)[0][:n]}
        out.append(np.stack([F.summarise(ff,lo,hi) for lo,hi in F.segment_bounds(len(ff['logrms']))]))
    return np.array(out,dtype=np.float32)

def train(p, seed, X, y, idx, out):
    torch.manual_seed(seed)
    model = LevelModel(p); model.mask_rng.manual_seed(seed+1000)
    opt = torch.optim.SGD(model.parameters(),lr=.02,momentum=.9,weight_decay=.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,60)
    history=[]; best=-1; t0=time.monotonic()
    for epoch in range(1,61):
        model.train(); perm=torch.randperm(len(idx['train'])); total=0.
        for b in perm.split(64):
            ii=idx['train'][b]
            loss=TF.cross_entropy(model(X[ii]),y[ii]); opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()*len(b)
        scheduler.step(); model.eval()
        with torch.no_grad():
            lg=model(X[idx['val']]); acc=float((lg.argmax(1)==y[idx['val']]).float().mean())
            vl=float(TF.cross_entropy(lg,y[idx['val']]))
        history.append({'epoch':epoch,'train_loss':total/len(perm),'val_loss':vl,'val_accuracy':acc})
        if acc>best:
            best=acc; best_epoch=epoch; state={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    name=f'C_p{p:g}_s{seed}'
    torch.save(state,out/f'{name}.pt')
    save(out/f'{name}.json',{'p':p,'seed':seed,'best_epoch':best_epoch,'parameters':sum(v.numel() for v in model.parameters()),
                           'seconds':time.monotonic()-t0,'history':history})
    print(name,'best epoch',best_epoch,'val',round(best,4),flush=True)
    return name,model

def metrics(prob,y):
    pred=prob.argmax(1); cm=confusion_matrix(y,pred,labels=range(6))
    return {'accuracy':float((pred==y).mean()),'macro_f1':float(f1_score(y,pred,average='macro')),
            'recall':dict(zip(F.EMOTIONS,(cm.diagonal()/cm.sum(1)).tolist())),
            'predicted_share':dict(zip(F.EMOTIONS,np.bincount(pred,minlength=6)/len(pred))),
            'confusion':cm.tolist()}

@torch.no_grad()
def probabilities(model,x,**kw):
    return torch.cat([model(b,**kw).softmax(1) for b in x.split(512)]).numpy()

def comparison(prob, base, y):
    r=metrics(prob,y)
    r.update(flip=float((prob.argmax(1)!=base.argmax(1)).mean()),
             probability_tv=float(np.abs(prob-base).sum(1).mean()/2),
             correct_to_wrong=float(((base.argmax(1)==y)&(prob.argmax(1)!=y)).mean()))
    return r

def report(models, X, y, idx, P, mean,std,out, stage):
    results={}; probs={}; ii=idx[stage]; yt=y[ii]
    # Measured gain effect relative to a matched freshly extracted clean signal.
    fresh=[torch.tensor((P[stage][:,v]-mean)/std) for v in range(3)]
    for name,model in models.items():
        model.eval()
        base=probabilities(model,X[ii]); pc=probabilities(model,fresh[0])
        probs[name]=base
        r={'clean':metrics(base,yt),'gain_clean':comparison(pc,base,yt),'gain':{}}
        for v,db in enumerate((-6,6),1):
            pp=probabilities(model,fresh[v]); r['gain'][str(db)]=comparison(pp,pc,yt)
            probs[f'{name}_gain{db}']=pp
        if isinstance(model,LevelModel):
            r['level_only_intervention']={}
            for db in (-6,6):
                shift=np.array([np.sqrt(128)*db,db*np.log(10)/20],dtype=np.float32)/std[LEVEL]
                pp=probabilities(model,X[ii],mu_shift=torch.tensor(shift))
                r['level_only_intervention'][str(db)]=comparison(pp,base,yt)
            with torch.no_grad():
                _,g,_=model.components(X[ii]); r['gate_mean']=float(g.mean())
            if stage=='val':
                r['strength_sweep']={}
                for strength in (0.,.25,.5,.75,1.):
                    b=probabilities(model,fresh[0],strength=strength)
                    c=[comparison(probabilities(model,fresh[v],strength=strength),b,yt) for v in (1,2)]
                    r['strength_sweep'][str(strength)]={'clean':metrics(probabilities(model,X[ii],strength=strength),yt),
                                                       'mean_gain_flip':float(np.mean([v['flip'] for v in c]))}
        results[name]=r
    save(out/f'{stage}.json',results)
    np.savez_compressed(out/f'{stage}-predictions.npz',idx=ii,y=yt,**probs)
    with open(out/f'{stage}.csv','w') as f:
        w=csv.writer(f);w.writerow(['model','accuracy','macro_f1',*F.EMOTIONS,'mean_gain_flip','mean_gain_tv'])
        for name,r in results.items():
            w.writerow([name,r['clean']['accuracy'],r['clean']['macro_f1'],*r['clean']['recall'].values(),
                        np.mean([v['flip'] for v in r['gain'].values()]),np.mean([v['probability_tv'] for v in r['gain'].values()])])
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',type=Path,required=True);ap.add_argument('--audio',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,default=ROOT/'results')
    args=ap.parse_args(); out=args.out
    if out.exists(): raise FileExistsError('Use a new output directory to preserve runs.')
    out.mkdir(parents=True);torch.set_num_threads(1)
    d=np.load(args.features); raw=d['X']; y=np.array([F.EMOTIONS.index(e) for e in d['emotion']]);idx,groups=split(d['speaker'])
    flat=raw[idx['train']].reshape(-1,88);mean=flat.mean(0);std=flat.std(0)+1e-6
    X=torch.tensor((raw-mean)/std);yt=torch.tensor(y)
    save(out/'config.json',{'feature_sha256':digest(args.features),'protocol_sha256':digest(ROOT/'protocol-zh.txt'),
                           'code_sha256':digest(__file__),'seeds':SEEDS,'p':PS,'groups':groups,
                           'torch':torch.__version__,'numpy':np.__version__,'features':str(args.features),'audio':str(args.audio)})
    np.savez_compressed(out/'normalization.npz',mean=mean,std=std)
    # Safe loading of known NumPy-only metadata alongside tensors in pilot checkpoints.
    torch.serialization.add_safe_globals([np.ndarray,np.dtype,np._core.multiarray._reconstruct,np.dtypes.Float32DType,np.dtypes.Float64DType])
    models={}
    for typ,tag in [('concat','A'),('concat_rel','B')]:
        for seed in SEEDS:
            ck=torch.load(args.pilot/f'{typ}_s{seed}'/'model.pt',weights_only=True,map_location='cpu')
            m=SegmentMLP(88,dropout=.5)
            state={k:v for k,v in ck['state_dict'].items() if k!='level_idx'}
            m.load_state_dict(state)
            if tag=='B': m=CenteredBaseline(m)
            # This cache was checked against pilot predictions in the separate audit.
            if not np.allclose(mean,ck['mean'],atol=1e-5) or not np.allclose(std,ck['std'],atol=1e-5):
                raise ValueError('Pilot normalization differs; use the same feature cache.')
            models[f'{tag}_s{seed}']=m
    # Waveform extraction and model fitting are independent; keep progress visible.
    P={}
    with mp.Pool(4) as pool:
        for stage in ('val','test'):
            paths=[str(args.audio/str(f)) for f in d['fname'][idx[stage]]]
            items=[]
            for n,arr in enumerate(pool.imap(wave_work,paths,chunksize=16),1):
                items.append(arr)
                if n%250==0: print('gain features',stage,n,'/',len(paths),flush=True)
            P[stage]=np.array(items)
    np.savez_compressed(out/'gain-features.npz',**P)
    for p in PS:
        for seed in SEEDS:
            name,m=train(p,seed,X,yt,idx,out);models[name]=m
    val=report(models,X,y,idx,P,mean,std,out,'val')
    selection=[]
    for p in PS:
        rs=[val[f'C_p{p:g}_s{s}'] for s in SEEDS]
        selection.append({'p':p,'val_accuracy':float(np.mean([r['clean']['accuracy'] for r in rs])),
                          'gain_flip':float(np.mean([v['flip'] for r in rs for v in r['gain'].values()]))})
    best=sorted(selection,key=lambda r:(-r['val_accuracy'],r['gain_flip'],-r['p']))[0]
    save(out/'selection.json',{'rule':'max mean validation clean accuracy; tie lower gain flip, then higher p',
                              'selected':best,'candidates':selection,'frozen_before_test':True})
    print('Validation decision frozen:',best,flush=True)
    report(models,X,y,idx,P,mean,std,out,'test')
    print('Finished',out,flush=True)

class CenteredBaseline(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,x):
        x=x.clone();x[:,:,LEVEL]-=x[:,:,LEVEL].mean(1,keepdim=True)
        return self.model(x)

if __name__=='__main__': main()
