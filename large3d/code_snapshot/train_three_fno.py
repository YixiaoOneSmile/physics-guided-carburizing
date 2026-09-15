"""Train matched target formulations with exact sample schedule and wall-time guard."""
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from large_fno import ROOT,Queries,model,loss_terms
from build_large_data import dump,check_deadline
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
@torch.no_grad()
def validate(net,queries):
    net.eval();total=0.;count=0;start=time.perf_counter()
    for row in queries.by_split["val"]:
        for q in range(21):
            s=queries.get(row,q,MODE)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                y=net(s["grid"],s["process"],s["query"])
                _,mse,sl1,_=loss_terms(y,s)
            total+=float(mse+.2*sl1);count+=1
    return total/count,time.perf_counter()-start
def main():
    global MODE
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["direct","prior_direct","residual"],required=True)
    ap.add_argument("--smoke",action="store_true");args=ap.parse_args();MODE=args.mode
    cfg=json.loads((ROOT/"configs"/"run.json").read_text());seed_all(cfg["seed"])
    if args.smoke:
        # Execution-only one-case smoke; never used as a scientific checkpoint.
        with __import__("h5py").File(ROOT/"data"/"train_000.h5") as f:
            acc=json.loads(f.attrs["stats"])
        stats={"carbon_mean":acc["carbon_sum"]/acc["count"],"carbon_std":.05,
               "residual_mean":acc["residual_sum"]/acc["count"],"residual_std":.01,
               "process_mean":[1.,900.,0.,0.],"process_std":[.1,30.,.1,.1]}
        queries=Queries(stats);epochs=1;steps=3
    else:queries=Queries();epochs=cfg["epochs"];steps=cfg["steps_per_epoch"]
    net=model();opt=torch.optim.AdamW(net.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    if args.smoke:
        sample=queries.get(queries.by_split["train"][0],10,MODE)
        net.eval()
        with torch.no_grad():
            fp=net(sample["grid"],sample["process"],sample["query"]).float()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                mp=net(sample["grid"],sample["process"],sample["query"]).float()
            rel=float(torch.linalg.vector_norm(mp-fp)/torch.linalg.vector_norm(fp).clamp_min(1e-6))
            mae=float((mp-fp).abs().mean())
        dump(ROOT/"results"/"mixed_precision_real_sample.json",{"relative_output_l2":rel,"normalized_mae":mae,
             "scope":"untrained real-input execution check, not final accuracy equivalence","passed":rel<.05})
        if rel>=.05:raise RuntimeError("Mixed precision real-input difference too large")
        del sample,fp,mp
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    rng=np.random.default_rng(cfg["seed"])
    cases=rng.integers(0,66,size=(epochs,steps,2));times=rng.integers(0,21,size=(epochs,steps,2))
    name=MODE+("_smoke" if args.smoke else "")
    folder=ROOT/"checkpoints"/name;folder.mkdir(exist_ok=True)
    if (folder/"last.pt").exists():raise RuntimeError("Existing checkpoint: inspect before resume")
    best=float("inf");started=time.time();validation_times=[];epoch_times=[]
    for epoch in range(epochs):
        check_deadline();net.train();sums=np.zeros(4);e0=time.perf_counter()
        for step in range(steps):
            check_deadline();opt.zero_grad(set_to_none=True)
            chosen=[queries.by_split["train"][0 if args.smoke else cases[epoch,step,mb]] for mb in range(2)]
            counts=[]
            for row in chosen:
                entry=queries.case(row);counts.append((int(entry["mask"].sum()),int(entry["surface"].sum())))
            for mb in range(2):
                row=chosen[mb]
                q=int(times[epoch,step,mb]);s=queries.get(row,q,MODE)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    y=net(s["grid"],s["process"],s["query"]);terms=loss_terms(y,s)
                if not torch.isfinite(terms[0]):raise RuntimeError("Nonfinite loss")
                # Exactly match old global masked batch means, not equal-case MSE.
                wm=counts[mb][0]/sum(c[0] for c in counts)
                ws=counts[mb][1]/max(sum(c[1] for c in counts),1)
                weighted=terms[1]*wm+.2*terms[2]*ws+.05*terms[3]/2
                weighted.backward();sums+=np.asarray([float(x.detach()) for x in terms])/2
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError("Nonfinite gradients")
            opt.step()
            if step%20==0:
                dump(ROOT/"results"/"training_progress.json",{"mode":name,"epoch":epoch+1,"step":step+1,
                     "epochs":epochs,"steps":steps,"elapsed_s":time.time()-started})
        epoch_s=time.perf_counter()-e0;epoch_times.append(epoch_s);scheduler.step()
        record={"mode":name,"epoch":epoch+1,"epoch_train_s":epoch_s,"elapsed_s":time.time()-started,
                "train_loss":sums[0]/steps,"train_mse":sums[1]/steps,"train_surface":sums[2]/steps}
        if not args.smoke and (epoch==0 or (epoch+1)%cfg["validation_interval"]==0 or epoch+1==epochs):
            val,vs=validate(net,queries);validation_times.append(vs)
            record.update(val_selection=val,val_s=vs)
            if val<best:
                best=val;torch.save({"model":net.state_dict(),"epoch":epoch+1,"mode":MODE,"cfg":cfg,
                     "stats":queries.stats,"best_val":best},folder/"best.pt")
        checkpoint={"model":net.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
                    "epoch":epoch+1,"mode":MODE,"cfg":cfg,"stats":queries.stats,"best_val":best,
                    "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                    "numpy_rng":np.random.get_state(),"python_rng":random.getstate()}
        torch.save(checkpoint,folder/"last.pt")
        with (ROOT/"logs"/(name+".jsonl")).open("a") as f:f.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        if not args.smoke and epoch==0:
            # Real HDF5/GPU end-to-end timing, not only synthetic GPU kernels.
            modes_left=3-cfg["modes"].index(MODE)
            estimate_one=epoch_s*epochs+(validation_times[-1] if validation_times else 0)*21
            estimate_remaining=estimate_one*modes_left-(time.time()-started)
            estimate_remaining+=((validation_times[-1] if validation_times else 0)*(22/12))*modes_left
            deadline=json.loads((ROOT/"configs"/"deadline.json").read_text())["deadline_epoch"]
            timing={"mode":MODE,"estimated_remaining_h":estimate_remaining/3600,
                    "remaining_budget_h":(deadline-time.time())/3600,"epoch_train_s":epoch_s,
                    "validation_s":validation_times[-1]}
            dump(ROOT/"results"/"actual_runtime_projection.json",timing)
            if estimate_remaining*1.1>deadline-time.time():
                raise RuntimeError("STOP_BUDGET: measured end-to-end projection exceeds remaining24h")
    queries.close()
    dump(ROOT/"results"/(name+"_training_complete.json"),{"mode":MODE,"elapsed_s":time.time()-started,"epochs":epochs})
if __name__=="__main__":main()
