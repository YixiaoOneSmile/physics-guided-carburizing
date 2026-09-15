"""Replay original analytic shapes and recompute full-scale dynamic/prior fields."""
import argparse,hashlib,json,math,time,os
from pathlib import Path
import h5py,numpy as np,torch
import geometry3d
from physical_gpu import coefficients,pcg
ROOT=Path(__file__).resolve().parents[1]
def dump(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def check_deadline():
    state=ROOT/"configs"/"deadline.json"
    if state.exists() and time.time()>json.loads(state.read_text())["deadline_epoch"]-120:
        raise RuntimeError("STOP_BUDGET: wall-clock deadline reached")
def cell_grid(n):
    a=(2*(np.arange(n,dtype=np.float32)+.5)/n-1)
    x,y,z=np.meshgrid(a,a,a,indexing="ij")
    return x,y,z,np.stack([x,y,z])
def geometry(row,n):
    rng=np.random.default_rng();rng.bit_generator.state=row["geometry_rng_state"]
    original=geometry3d.coordinate_grid;geometry3d.coordinate_grid=cell_grid
    geom=geometry3d.make_geometry(n,row["shape_name"],rng)
    geometry3d.coordinate_grid=original
    mask=torch.as_tensor(geom.mask>0,device="cuda")
    faces=torch.as_tensor(geom.surface.sum(0),device="cuda")
    pairs=[]
    for ax in range(3):
        s=[slice(None)]*3;t=s.copy();s[ax]=slice(0,-1);t[ax]=slice(1,None)
        s,t=tuple(s),tuple(t);pairs.append((s,t,mask[s]&mask[t]))
    # EDT originally uses node spacing 2/(n-1); convert to cell spacing 2/n.
    return mask,faces,pairs,geom.sdf*((n-1)/n)
def solve(row,n,mask,faces,pairs,callback,avg=False,dtmax=60):
    mat=row["material"];c0=mat[0];dx=.05/n
    hist=np.asarray(row["process"]);ts=np.asarray(row["time_s"]);cp=hist[0].copy()
    if avg:cp[:]=np.trapezoid(cp,ts)/(ts[-1]-ts[0])
    u=torch.zeros(mask.shape,device="cuda",dtype=torch.float32)
    callback(0,torch.where(mask,u+c0,0))
    max_mismatch=0.;maxiter=0;steps=0;t0=time.perf_counter()
    for j in range(1,len(ts)):
        check_deadline()
        count=math.ceil((ts[j]-ts[j-1])/dtmax);dt=(ts[j]-ts[j-1])/count
        for k in range(count):
            tm=ts[j-1]+(k+.5)*dt;temp=float(np.interp(tm,ts,hist[1]))
            eq=float(np.clip(np.interp(tm,ts,cp)*(1-1.5e-4*(temp-930)),.02,1.35))
            diag,bc,coeff=coefficients(u,dt,temp,mat,dx,faces,pairs)
            new,it,rel=pcg(u+bc*(eq-c0),u,diag,coeff,pairs)
            balance=((new-u).sum(dtype=torch.float64)-(bc*(eq-c0-new)).sum(dtype=torch.float64)).abs()
            max_mismatch=max(max_mismatch,float(balance))
            u=new;maxiter=max(maxiter,it);steps+=1
        field=torch.where(mask,u+c0,0)
        if not torch.isfinite(field).all():raise RuntimeError("Non-finite physical solution")
        if float(field[mask].min())<c0-1e-5:raise RuntimeError("Unexpected sub-initial concentration")
        callback(j,field)
    uptake=float(u.sum(dtype=torch.float64))
    return {"steps":steps,"seconds_including_callbacks":time.perf_counter()-t0,
            "max_mass_mismatch_relative_to_total_uptake":max_mismatch/max(abs(uptake),1e-20),
            "max_pcg_iterations":maxiter}
def one(row,n):
    target=ROOT/"data"/(row["id"]+".h5")
    if target.exists():
        with h5py.File(target) as f:
            assert bool(f.attrs.get("complete",False))
            return json.loads(f.attrs["stats"]),json.loads(f.attrs["timings"])
    started=time.perf_counter();mask,faces,pairs,sdf=geometry(row,n)
    stats={"count":0,"carbon_sum":0.,"carbon_sq_sum":0.,"residual_sum":0.,"residual_sq_sum":0.}
    tmp=target.with_suffix(".partial.h5")
    if tmp.exists(): raise RuntimeError(f"Partial output exists; inspect before resuming {tmp}")
    with h5py.File(tmp,"w") as f:
        f.attrs["metadata"]=json.dumps(row);f.attrs["n"]=n;f.attrs["domain_mm"]=50.
        for key,data in (("mask",mask.cpu().numpy()),("surface_count",faces.cpu().numpy().astype(np.uint8)),("sdf",sdf)):
            f.create_dataset(key,data=data,compression="gzip",compression_opts=1,shuffle=True)
        nt=len(row["time_s"])
        for key in ("dynamic","average"):
            f.create_dataset(key,shape=(nt,n,n,n),dtype="f4",chunks=(1,32,128,128),
                             compression="gzip",compression_opts=1,shuffle=True)
        def dynamic(q,field):f["dynamic"][q]=field.cpu().numpy()
        dyn=solve(row,n,mask,faces,pairs,dynamic)
        def average(q,field):
            f["average"][q]=field.cpu().numpy()
            if row["split"]=="train":
                truth=torch.as_tensor(f["dynamic"][q],device="cuda")[mask].double()
                res=truth-field[mask].double()
                stats["count"]+=truth.numel()
                stats["carbon_sum"]+=float(truth.sum());stats["carbon_sq_sum"]+=float(truth.square().sum())
                stats["residual_sum"]+=float(res.sum());stats["residual_sq_sum"]+=float(res.square().sum())
        avg=solve(row,n,mask,faces,pairs,average,avg=True)
        timings={"dynamic":dyn,"average":avg,"total_s":time.perf_counter()-started}
        if max(dyn["max_mass_mismatch_relative_to_total_uptake"],avg["max_mass_mismatch_relative_to_total_uptake"])>1e-4:
            raise RuntimeError("Mass conservation gate failed")
        f.attrs["stats"]=json.dumps(stats);f.attrs["timings"]=json.dumps(timings);f.attrs["complete"]=True
    tmp.replace(target)
    return stats,timings
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--limit",type=int,default=0);args=ap.parse_args()
    torch.set_num_threads(2);cfg=json.loads((ROOT/"configs"/"run.json").read_text())
    rows=json.loads((ROOT/"configs"/"cases.json").read_text())["rows"]
    if args.limit:rows=rows[:args.limit]
    totals={"count":0,"carbon_sum":0.,"carbon_sq_sum":0.,"residual_sum":0.,"residual_sq_sum":0.}
    times=[];start=time.time()
    for i,row in enumerate(rows):
        check_deadline();stats,timing=one(row,cfg["n"])
        for key in totals:totals[key]+=stats[key]
        times.append(timing["total_s"])
        record={"completed":i+1,"total":len(rows),"case":row["id"],"last_case_s":timing["total_s"],
                "estimated_remaining_data_h":float(np.mean(times))*(len(rows)-i-1)/3600,"elapsed_s":time.time()-start}
        dump(ROOT/"results"/"data_progress.json",record);print(json.dumps(record),flush=True)
    if not args.limit:
        count=totals["count"];out={"active_value_count":count,"train_only":True}
        for name in ("carbon","residual"):
            mean=totals[name+"_sum"]/count;var=totals[name+"_sq_sum"]/count-mean**2
            out[name+"_mean"]=mean;out[name+"_std"]=max(math.sqrt(max(var,0)),1e-7)
        processes=[]
        for row in rows:
            if row["split"]!="train":continue
            ts=np.asarray(row["time_s"]);hist=np.asarray(row["process"]);cp=hist[0]
            avg=np.trapezoid(cp,ts)/(ts[-1]-ts[0]);centered=cp-avg
            cum=np.r_[0,np.cumsum(.5*(centered[:-1]+centered[1:])*np.diff(ts))]/ts[-1]
            processes.append(np.stack([cp,hist[1],centered,cum]))
        process=np.stack(processes)
        out["process_mean"]=process.mean((0,2)).tolist();out["process_std"]=np.maximum(process.std((0,2)),1e-7).tolist()
        dump(ROOT/"configs"/"stats.json",out)
        dump(ROOT/"results"/"data_complete.json",{"cases":len(rows),"elapsed_s":time.time()-start,"stats":out})
if __name__=="__main__":main()
