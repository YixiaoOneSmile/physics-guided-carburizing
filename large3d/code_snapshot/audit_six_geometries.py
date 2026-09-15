"""Representative 256->512 checks for all six replayed analytic families."""
import json,time
import h5py,numpy as np,torch
import torch.nn.functional as F
from build_large_data import ROOT,geometry,solve,one,dump,check_deadline
def main():
    torch.set_num_threads(2)
    rows=json.loads((ROOT/"configs"/"cases.json").read_text())["rows"]
    selected=[]
    for sid in range(6):selected.append(next(r for r in rows if r["split"]=="train" and r["shape_id"]==sid))
    results=[]
    for row in selected:
        check_deadline();one(row,256)
        with h5py.File(ROOT/"data"/(row["id"]+".h5")) as f:
            coarse=torch.as_tensor(f["dynamic"][-1],device="cuda")
            cmask=torch.as_tensor(f["mask"][:],device="cuda")
            sdf=torch.as_tensor(f["sdf"][:],device="cuda")
        mask,faces,pairs,_=geometry(row,512);last=[]
        def callback(q,field):
            if q==20:last.append(field)
        timing=solve(row,512,mask,faces,pairs,callback)
        fine=last[0];fraction=F.avg_pool3d(mask.float()[None,None],2)[0,0]
        averaged=F.avg_pool3d(fine[None,None],2)[0,0]/fraction.clamp_min(.125)
        common=cmask&(fraction>=.5);shell=common&(sdf<=.12)
        mae=float((coarse-averaged).abs()[shell].mean())
        c0=row["material"][0]
        coarse_uptake=float(((coarse-c0)*cmask).sum(dtype=torch.float64))/256**3
        fine_uptake=float(((fine-c0)*mask).sum(dtype=torch.float64))/512**3
        volume_rel=abs(float(cmask.sum())/256**3-float(mask.sum())/512**3)/(float(mask.sum())/512**3)
        uptake_rel=abs(coarse_uptake-fine_uptake)/abs(fine_uptake)
        result={"case":row["id"],"shape":row["shape_name"],"shell_final_mae_wt":mae,
                "volume_relative_change":volume_rel,"total_uptake_relative_change":uptake_rel,
                "comparison_fraction_of_coarse_material":float(common.sum()/cmask.sum()),
                "fine_solver":timing,"passed":mae<.005 and volume_rel<.03 and uptake_rel<.05}
        np.savez_compressed(ROOT/"results"/(row["shape_name"]+"_mesh_audit.npz"),
            coarse_section=coarse[:,:,128].cpu().numpy(),fine_section=fine[:,:,256].cpu().numpy(),
            coarse_mask=cmask[:,:,128].cpu().numpy(),fine_mask=mask[:,:,256].cpu().numpy())
        results.append(result);dump(ROOT/"results"/"six_geometry_audit.json",{"cases":results,"all_passed":all(r["passed"] for r in results)})
        print(json.dumps(result),flush=True)
        del coarse,cmask,sdf,mask,faces,pairs,last,fine,averaged,fraction,common,shell
        torch.cuda.empty_cache()
    assert len(results)==6 and all(r["passed"] for r in results),"STOP_NUMERICS: one geometry failed refinement gates"
if __name__=="__main__":main()
