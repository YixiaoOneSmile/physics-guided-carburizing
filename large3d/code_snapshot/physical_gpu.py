"""Conservative full 3-D Cartesian FV pilot. No symmetry reduction.

New discretization: cell-centred, harmonic internal conductances, half-cell
diffusion resistance in Robin condition, no clipping. Old labels untouched.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

def geometry(n,device,dtype):
    a=(torch.arange(n,device=device,dtype=dtype)+0.5)/n-0.5
    x,y,z=a[:,None,None],a[None,:,None],a[None,None,:]
    mask=~((x>0.25)&(y.abs()<0.125)&(z.abs()<0.25))
    mask=mask.expand(n,n,n).contiguous()
    faces=torch.zeros((n,n,n),device=device,dtype=dtype)
    pairs=[]
    for ax in range(3):
        s=[slice(None)]*3;t=[slice(None)]*3;s[ax]=slice(0,-1);t[ax]=slice(1,None)
        s,t=tuple(s),tuple(t)
        pairs.append((s,t,mask[s]&mask[t]))
        faces[s]+=(mask[s]&~mask[t]).to(dtype)
        faces[t]+=(mask[t]&~mask[s]).to(dtype)
        q=[slice(None)]*3;q[ax]=0;faces[tuple(q)]+=mask[tuple(q)]
        q[ax]=-1;faces[tuple(q)]+=mask[tuple(q)]
    return mask,faces,pairs

def coefficients(u,dt,temp,mat,dx,faces,pairs):
    c0,hm,dref,Q=mat[:4]
    D=dref*math.exp(-Q/8.31446261815324*(1/(temp+273.15)-1/1203.15))*(1+.18*(u+c0))
    bc=(dt/dx)*faces/(1/hm+dx/(2*D))
    diag=1+bc
    coeff=[]
    for s,t,active in pairs:
        f=(dt/dx**2)*(2*D[s]*D[t]/(D[s]+D[t]))*active
        diag[s]+=f;diag[t]+=f;coeff.append(f)
    return diag,bc,coeff

def matvec(v,diag,coeff,pairs):
    out=diag*v
    for f,(s,t,_) in zip(coeff,pairs):
        out[s]-=f*v[t];out[t]-=f*v[s]
    return out

def pcg(rhs,x,diag,coeff,pairs,rtol=2e-7,maxiter=150):
    x=x.clone()
    r=rhs-matvec(x,diag,coeff,pairs)
    bn=torch.linalg.vector_norm(rhs)
    if float(bn)==0: return x.zero_(),0,0.
    z=r/diag;p=z.clone();rz=torch.sum(r*z)
    for k in range(maxiter):
        ap=matvec(p,diag,coeff,pairs)
        alpha=rz/torch.sum(p*ap)
        x.add_(alpha*p);r.sub_(alpha*ap)
        rel=float(torch.linalg.vector_norm(r)/bn)
        if rel<rtol: return x,k+1,rel
        z=r/diag;new=torch.sum(r*z)
        p=z+(new/rz)*p;rz=new
    raise RuntimeError(f"PCG did not converge: {rel}")

def audit(root,cfg):
    n=8;dx=.05/n;mat=cfg["material"]
    mask,faces,pairs=geometry(n,"cuda",torch.float64)
    torch.manual_seed(2027)
    u=torch.rand((n,n,n),device="cuda",dtype=torch.float64)*.1*mask
    diag,bc,coeff=coefficients(u,60.,916.,mat,dx,faces,pairs)
    rhs=u+bc*.7
    gpu,it,rel=pcg(rhs,u,diag,coeff,pairs,rtol=1e-12)
    ids=np.arange(n**3).reshape(n,n,n)
    rows=[ids.ravel()];cols=[ids.ravel()];vals=[diag.cpu().numpy().ravel()]
    for f,(s,t,_) in zip(coeff,pairs):
        v=-f.cpu().numpy().ravel()
        rows.extend([ids[s].ravel(),ids[t].ravel()])
        cols.extend([ids[t].ravel(),ids[s].ravel()]);vals.extend([v,v])
    A=sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(n**3,n**3)).tocsr()
    cpu=spsolve(A,rhs.cpu().numpy().ravel()).reshape(n,n,n)
    delta=(gpu-u).sum().item()*dx**3
    flux=(bc*(.7-gpu)).sum().item()*dx**3
    zero=torch.zeros_like(u)
    control,_,_=pcg(zero,zero,diag,coeff,pairs)
    out={"gpu_cpu_max_abs":float(np.max(np.abs(cpu-gpu.cpu().numpy()))),
         "linear_residual":rel,"iterations":it,
         "relative_mass_balance":abs(delta-flux)/abs(delta),
         "constant_control_max_abs":float(control.abs().max()),
         "active_cells":int(mask.sum()),"expected_volume_m3":.05**3-.0125*.0125*.025,
         "voxel_volume_m3":float(mask.sum())*dx**3}
    out["passed"]=out["gpu_cpu_max_abs"]<1e-10 and out["relative_mass_balance"]<1e-9 and out["constant_control_max_abs"]==0
    (root/"results"/"gpu_solver_audit.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)
    assert out["passed"]

def run(root,cfg,n,dtmax,average=False,stop_after=0):
    dtype=torch.float32;device="cuda";dx=.05/n
    mat=cfg["material"];c0=mat[0]
    ts=np.asarray(cfg["time_s"]);hist=np.asarray(cfg["process"])
    cp=hist[0].copy()
    if average: cp[:]=np.trapezoid(cp,ts)/(ts[-1]-ts[0])
    name=f"notched_n{n}_dt{int(dtmax)}"+("_average" if average else "")
    mask,faces,pairs=geometry(n,device,dtype)
    u=torch.zeros((n,n,n),device=device,dtype=dtype)
    # Full model domain; no collapsed dimension.
    time_out=[0.];surface_hist=[c0];inventory=[float(mask.sum())*c0*dx**3]
    sections=[np.where(mask[:,:,n//2].cpu().numpy(),c0,np.nan)]
    profile=[(u[:,n//2,n//2]+c0).cpu().numpy()]
    torch.cuda.synchronize();started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    max_balance=0.;max_iter=0;count=0;worst_rel=0.;last_temp=float(hist[1,0])
    for j in range(1,len(ts)):
        steps=int(math.ceil((ts[j]-ts[j-1])/dtmax));dt=(ts[j]-ts[j-1])/steps
        for k in range(steps):
            tm=ts[j-1]+(k+.5)*dt
            temp=float(np.interp(tm,ts,hist[1]))
            eq=float(np.clip(np.interp(tm,ts,cp)*(1-1.5e-4*(temp-930)),.02,1.35))
            diag,bc,coeff=coefficients(u,dt,temp,mat,dx,faces,pairs)
            rhs=u+bc*(eq-c0)
            new,it,rel=pcg(rhs,u,diag,coeff,pairs)
            delta=(new-u).sum(dtype=torch.float64)*dx**3
            flux=(bc*(eq-c0-new)).sum(dtype=torch.float64)*dx**3
            mismatch=float((delta-flux).abs())
            max_balance=max(max_balance,mismatch)
            u=new;count+=1;max_iter=max(max_iter,it);worst_rel=max(worst_rel,rel);last_temp=temp
            if stop_after and count>=stop_after: break
        elapsed=time.perf_counter()-started
        at=ts[j] if not stop_after else ts[j-1]+(k+1)*dt
        t_eq=float(np.clip(np.interp(at,ts,cp)*(1-1.5e-4*(np.interp(at,ts,hist[1])-930)),.02,1.35))
        D=mat[2]*math.exp(-mat[3]/8.31446261815324*(1/(last_temp+273.15)-1/1203.15))*(1+.18*float(u[0,n//2,n//2]+c0))
        cc=float(u[0,n//2,n//2]+c0)
        cs=(mat[1]*t_eq+(2*D/dx)*cc)/(mat[1]+2*D/dx)
        time_out.append(float(at));surface_hist.append(cs)
        inventory.append(float((u*mask).sum(dtype=torch.float64)+c0*mask.sum())*dx**3)
        sections.append(torch.where(mask[:,:,n//2],u[:,:,n//2]+c0,torch.nan).cpu().numpy())
        profile.append((u[:,n//2,n//2]+c0).cpu().numpy())
        print(json.dumps({"name":name,"output":j,"steps":count,"elapsed_s":elapsed,"max_cg":max_iter}),flush=True)
        if stop_after and count>=stop_after: break
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    field=torch.where(mask,u+c0,0).cpu().numpy()
    np.save(root/"results"/f"{name}_final.npy",field)
    np.savez_compressed(root/"results"/f"{name}_history.npz",time=time_out,surface=surface_hist,
                        inventory=inventory,section=np.asarray(sections),profile=np.asarray(profile),
                        x_mm=(np.arange(n)+.5)*dx*1000-25)
    uptake=inventory[-1]-inventory[0]
    out={"name":name,"n":n,"dx_mm":dx*1000,"dt_max_s":dtmax,"steps":count,
         "elapsed_s":elapsed,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
         "active_cells":int(mask.sum()),"relative_mass_mismatch_to_total_uptake":max_balance/max(abs(uptake),1e-30),
         "max_pcg_iterations":max_iter,"worst_pcg_relative_residual":worst_rel,
         "completed_full_process":not stop_after,"average_cp":average,"c_min":float(field[field>0].min()),
         "c_max":float(field.max()),"dtype":"float32"}
    (root/"results"/f"{name}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--n",type=int,default=128);ap.add_argument("--dt",type=float,default=60)
    ap.add_argument("--audit",action="store_true");ap.add_argument("--average",action="store_true")
    ap.add_argument("--stop-after",type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(2)
    cfg=json.loads((args.root/"configs"/"source_case.json").read_text())
    if args.audit: audit(args.root,cfg)
    else: run(args.root,cfg,args.n,args.dt,args.average,args.stop_after)
if __name__=="__main__":main()
