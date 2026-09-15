"""Separate PDE discretization from changing voxelized shape boundaries.

The 256^3 voxel material domain is the model input and label domain. Subdivide
those exact cells at 512^3: same material volume and exposed staircase geometry,
not an independently re-voxelized analytic shape. This tests discretization only.
The independent geometry-refinement discrepancy is retained separately.
"""
import hashlib,json
import h5py,numpy as np,torch
import torch.nn.functional as F
from build_large_data import ROOT,geometry,solve,one,dump,check_deadline
def topology(mask):
    faces=torch.zeros(mask.shape,device="cuda")
    pairs=[]
    for ax in range(3):
        s=[slice(None)]*3;t=s.copy();s[ax]=slice(0,-1);t[ax]=slice(1,None);s,t=tuple(s),tuple(t)
        pairs.append((s,t,mask[s]&mask[t]))
        faces[s]+=(mask[s]&~mask[t]);faces[t]+=(mask[t]&~mask[s])
        q=[slice(None)]*3;q[ax]=0;faces[tuple(q)]+=mask[tuple(q)]
        q[ax]=-1;faces[tuple(q)]+=mask[tuple(q)]
    return faces,pairs
def main():
    torch.set_num_threads(2);rows=json.loads((ROOT/"configs"/"cases.json").read_text())["rows"]
    selected=[next(r for r in rows if r["split"]=="train" and r["shape_id"]==sid) for sid in range(6)]
    results=[]
    for row in selected:
        check_deadline();one(row,256)
        with h5py.File(ROOT/"data"/(row["id"]+".h5")) as f:
            coarse=torch.as_tensor(f["dynamic"][-1],device="cuda")
            cmask=torch.as_tensor(f["mask"][:],device="cuda")
            sdf=torch.as_tensor(f["sdf"][:],device="cuda")
        mask=cmask.repeat_interleave(2,0).repeat_interleave(2,1).repeat_interleave(2,2)
        faces,pairs=topology(mask);last=[]
        def callback(q,field):
            if q==20:last.append(field)
        timing=solve(row,512,mask,faces,pairs,callback)
        fine=last[0];averaged=F.avg_pool3d(fine[None,None],2)[0,0]
        assert int(mask.sum())==int(cmask.sum())*8
        shell=cmask&(sdf<=.12);error=(coarse-averaged).abs()
        mae=float(error[shell].mean());c0=row["material"][0]
        cu=float(((coarse-c0)*cmask).sum(dtype=torch.float64))/256**3
        fu=float(((fine-c0)*mask).sum(dtype=torch.float64))/512**3
        uptake_rel=abs(cu-fu)/abs(fu)
        result={"case":row["id"],"shape":row["shape_name"],"reference":"exact subdivision of the same voxel material domain",
                "shell_final_mae_wt":mae,"active_final_mae_wt":float(error[cmask].mean()),
                "volume_relative_change":0.,"total_uptake_relative_change":uptake_rel,
                "fine_solver":timing,"passed":mae<.005 and uptake_rel<.05,
                "does_not_validate":"continuum analytic-shape boundary approximation"}
        np.savez_compressed(ROOT/"results"/(row["shape_name"]+"_fixed_mesh_audit.npz"),
            coarse_section=coarse[:,:,128].cpu().numpy(),fine_section=fine[:,:,256].cpu().numpy(),
            coarse_mask=cmask[:,:,128].cpu().numpy(),fine_mask=mask[:,:,256].cpu().numpy())
        results.append(result)
        dump(ROOT/"results"/"fixed_geometry_audit.json",{"cases":results,"all_passed":all(r["passed"] for r in results)})
        print(json.dumps(result),flush=True)
        if not result["passed"]:raise RuntimeError("STOP_NUMERICS: PDE refinement failed even on fixed geometry")
        del coarse,cmask,sdf,mask,faces,pairs,last,fine,averaged,error,shell
        torch.cuda.empty_cache()
    dump(ROOT/"results"/"fixed_geometry_audit_complete.json",{"cases":6,"all_passed":True})
if __name__=="__main__":main()
