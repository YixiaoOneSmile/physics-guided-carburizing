"""All21 times on all22 test cases; primary metrics match the small-scale matrix."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from large_fno import ROOT,Queries,model,physical
from build_large_data import dump,check_deadline
METRICS=["active_field_rel_l2","active_mae_wt","surface_c_mae_wt","final_active_mae_wt",
         "mass_uptake_rel_error","shell_mae_wt","raw_active_mae_wt","out_of_bounds_fraction"]
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",required=True);args=ap.parse_args()
    torch.set_num_threads(2);queries=Queries();net=model()
    ck=torch.load(ROOT/"checkpoints"/args.mode/"best.pt",map_location="cuda",weights_only=False)
    net.load_state_dict(ck["model"]);net.eval();records=[];prior_records=[]
    started=time.time();out=ROOT/"results"/args.mode;out.mkdir(exist_ok=True)
    for row in queries.by_split["test"]:
        sums=[dict(se=0.,y2=0.,ae=0.,sae=0.,shae=0.,count=0,scount=0,shcount=0,mass=0.,raw=0.,bad=0) for _ in range(2)]
        sections=[]
        for q in range(21):
            check_deadline();s=queries.get(row,q,args.mode)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                y=net(s["grid"],s["process"],s["query"])
            raw=physical(y,s,queries.stats,args.mode)
            for acc,pred,rawpred in zip(sums,(raw.clamp(.02,1.45),s["prior"]),(raw,s["prior"])):
                mask=s["mask"];surface=s["surface"];shell=s["shell"];truth=s["truth"]
                diff=pred-truth
                acc["se"]+=float(diff[mask].double().square().sum());acc["y2"]+=float(truth[mask].double().square().sum())
                acc["ae"]+=float(diff[mask].abs().double().sum());acc["count"]+=int(mask.sum())
                acc["sae"]+=float(diff[surface].abs().double().sum());acc["scount"]+=int(surface.sum())
                acc["shae"]+=float(diff[shell].abs().double().sum());acc["shcount"]+=int(shell.sum())
                acc["mass"]+=float(diff[mask].double().sum().abs()/truth[mask].double().sum().abs().clamp_min(1e-12))
                acc["raw"]+=float((rawpred-truth)[mask].abs().double().sum())
                acc["bad"]+=int((((rawpred<.02)|(rawpred>1.45))&mask).sum())
                if q==20:acc["final"]=float(diff[mask].abs().mean())
            sections.append(raw[0,0,:,:,queries.n//2].cpu().numpy())
        for dest,acc in zip((records,prior_records),sums):
            dest.append({"case":row["id"],"shape":row["shape_name"],
                "active_field_rel_l2":float(np.sqrt(acc["se"]/acc["y2"])),
                "active_mae_wt":acc["ae"]/acc["count"],"surface_c_mae_wt":acc["sae"]/acc["scount"],
                "final_active_mae_wt":acc["final"],"mass_uptake_rel_error":acc["mass"]/21,
                "shell_mae_wt":acc["shae"]/acc["shcount"],"raw_active_mae_wt":acc["raw"]/acc["count"],
                "out_of_bounds_fraction":acc["bad"]/acc["count"]})
        np.savez_compressed(out/(row["id"]+"_sections.npz"),raw_prediction=np.stack(sections))
        print(json.dumps({"mode":args.mode,"case":row["id"],"done":len(records),"total":22}),flush=True)
    def summarize(rows):return {k:float(np.mean([r[k] for r in rows])) for k in METRICS}
    result={"mode":args.mode,"seed":2027,"checkpoint_epoch":ck["epoch"],"test_cases":22,
            "times_per_case":21,"prediction":summarize(records),"average_cp":summarize(prior_records),
            "by_shape":{sh:summarize([r for r in records if r["shape"]==sh]) for sh in sorted(set(r["shape"] for r in records))},
            "cases":records,"prior_cases":prior_records,"elapsed_s":time.time()-started,
            "mass_metric_note":"relative total carbon inventory error; original key retained"}
    dump(out/"evaluation.json",result)
    with (out/"case_metrics.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(result["prediction"]),flush=True);queries.close()
if __name__=="__main__":main()
