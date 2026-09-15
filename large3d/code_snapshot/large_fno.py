"""Memory-bounded query construction and unchanged FNO with fp32 spectral kernels."""
import json
from collections import OrderedDict
from pathlib import Path
from types import MethodType
import h5py,numpy as np,torch
from matrix_models3d import MatrixPredictor3D
ROOT=Path(__file__).resolve().parents[1]
def model():
    net=MatrixPredictor3D(backbone="fno",grid_channels=17).cuda()
    for name,module in net.named_modules():
        if "SpectralConv" in type(module).__name__:
            original=module.forward
            def safe(self,x,_original=original):
                with torch.autocast("cuda",enabled=False): y=_original(x.float())
                return y.to(x.dtype)
            module.forward=MethodType(safe,module)
    return net
class Queries:
    def __init__(self,stats=None):
        self.cfg=json.loads((ROOT/"configs"/"run.json").read_text());self.n=self.cfg["n"]
        self.stats=stats or json.loads((ROOT/"configs"/"stats.json").read_text())
        self.rows=json.loads((ROOT/"configs"/"cases.json").read_text())["rows"]
        self.by_split={s:[r for r in self.rows if r["split"]==s] for s in ("train","val","test")}
        self.cache=OrderedDict()
        a=2*(torch.arange(self.n,device="cuda",dtype=torch.float32)+.5)/self.n-1
        self.coords=torch.stack(torch.meshgrid(a,a,a,indexing="ij"))
    def case(self,row):
        key=row["id"]
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        if len(self.cache)>=2:
            _,old=self.cache.popitem(last=False);old["f"].close()
        f=h5py.File(ROOT/"data"/(key+".h5"),"r",rdcc_nbytes=4*1024**2)
        assert f.attrs["complete"]
        mask=torch.as_tensor(f["mask"][:],device="cuda")[None,None]
        sdf=torch.as_tensor(f["sdf"][:],device="cuda")[None,None]
        faces=torch.as_tensor(f["surface_count"][:],device="cuda",dtype=torch.float32)[None,None]
        hist=np.asarray(row["process"],dtype=np.float64);ts=np.asarray(row["time_s"])
        cp=hist[0];mean=np.trapezoid(cp,ts)/ts[-1];centered=cp-mean
        cum=np.r_[0,np.cumsum(.5*(centered[:-1]+centered[1:])*np.diff(ts))]/ts[-1]
        proc=np.stack([cp,hist[1],centered,cum])
        proc=(proc-np.asarray(self.stats["process_mean"])[:,None])/np.asarray(self.stats["process_std"])[:,None]
        entry={"f":f,"mask":mask,"sdf":sdf,"faces":faces,"surface":(faces>0)&mask,
               "shell":(sdf<=.12)&mask,"process":torch.as_tensor(proc,device="cuda",dtype=torch.float32)[None],
               "cp_mean":mean}
        self.cache[key]=entry;return entry
    def get(self,row,q,mode):
        c=self.case(row);s=self.stats
        truth=torch.as_tensor(c["f"]["dynamic"][q],device="cuda")[None,None]
        prior=torch.as_tensor(c["f"]["average"][q],device="cuda")[None,None]
        grid=torch.empty((1,17,self.n,self.n,self.n),device="cuda")
        grid[:,0:1]=c["mask"];grid[:,1:2]=c["sdf"];grid[:,2:3]=c["faces"]/6
        grid[:,3:6]=self.coords
        cp,temp=np.asarray(row["process"]);ts=np.asarray(row["time_s"])
        c0,hm,dref,Q,_=row["material"]
        vals=[ts[q]/ts[-1],c0,np.log10(hm),np.log10(dref),Q/150000.,c["cp_mean"],
              np.std(cp),np.trapezoid(cp[:q+1],ts[:q+1])/ts[-1],cp[q],(temp[q]-900)/60]
        for channel,value in enumerate(vals,6):grid[:,channel].fill_(float(value))
        if mode=="direct":grid[:,16].zero_()
        else:grid[:,16:17]=(prior-s["carbon_mean"])/s["carbon_std"]
        if mode=="residual":target=(truth-prior-s["residual_mean"])/s["residual_std"]
        else:target=(truth-s["carbon_mean"])/s["carbon_std"]
        return {"grid":grid,"process":c["process"],"query":torch.tensor([q],device="cuda"),
                "target":target,"truth":truth,"prior":prior,**{k:c[k] for k in ("mask","surface","shell")}}
    def close(self):
        for c in self.cache.values():c["f"].close()
        self.cache.clear()
def loss_terms(y,s):
    diff=y.float()-s["target"];mask=s["mask"];surface=s["surface"]
    mse=(diff.square()*mask).sum()/mask.sum()
    sl1=(diff.abs()*surface).sum()/surface.sum().clamp_min(1)
    ml1=((diff*mask).sum()/mask.sum()).abs()
    return mse+.2*sl1+.05*ml1,mse,sl1,ml1
def physical(y,s,stats,mode):
    if mode=="residual":return s["prior"]+y.float()*stats["residual_std"]+stats["residual_mean"]
    return y.float()*stats["carbon_std"]+stats["carbon_mean"]
