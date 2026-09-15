import json,time
from pathlib import Path
from build_large_data import ROOT,dump
modes=["direct","prior_direct","residual"]
reports={m:json.loads((ROOT/"results"/m/"evaluation.json").read_text()) for m in modes}
prior=reports["residual"]["average_cp"];res=reports["residual"]["prediction"]
def gain(base,key):return 100*(1-res[key]/base[key]) if base[key]!=0 else None
keys=["active_field_rel_l2","active_mae_wt","surface_c_mae_wt","final_active_mae_wt","mass_uptake_rel_error","shell_mae_wt"]
comparison={"status":"complete","scope":"single-seed large-scale retraining, not zero-shot transfer",
 "models":{m:reports[m]["prediction"] for m in modes},"average_cp":prior,
 "residual_improvement_percent":{m:{k:gain(reports[m]["prediction"],k) for k in keys} for m in modes[:2]},
 "residual_vs_average_percent":{k:gain(prior,k) for k in keys},
 "primary_improves_vs_prior_direct":res["active_field_rel_l2"]<reports["prior_direct"]["prediction"]["active_field_rel_l2"]}
dump(ROOT/"results"/"FINAL_COMPARISON.json",comparison)
lines=["# 大尺度FNO三组目标实验结果","",
 "范围：50 mm计算域，256³网格，原100例工艺与解析几何精确复现，66/12/22划分；每组一个随机种子2027。",
 "这是大尺度数据重新训练后的补充验证，不是小尺度模型直接跨尺度泛化。","",
 "| 方法 | 有效区相对L2 | 有效区MAE (wt%) | 表面MAE (wt%) | 终态MAE (wt%) |",
 "|---|---:|---:|---:|---:|"]
for label,v in [("Average-Cp",prior)]+[(m,reports[m]["prediction"]) for m in modes]:
    lines.append(f"| {label} | {v['active_field_rel_l2']:.6f} | {v['active_mae_wt']:.6f} | {v['surface_c_mae_wt']:.6f} | {v['final_active_mae_wt']:.6f} |")
g=gain(reports["prior_direct"]["prediction"],"active_field_rel_l2")
lines+=["",f"残差目标相对prior-direct的主指标改善为 {g:.2f}%（负值表示变差）。",
 "完整分形状、逐算例、未裁剪指标、近表层指标见 results/。仅一个种子，不据此宣称训练稳定性或未知形状泛化。",
 "模型选择仅使用验证集；旧小尺度数据、检查点和中英文论文未改。",
 "固定体素材料域的网格加密检查与重新体素化的几何敏感性分别记录。后者平板曾未通过门槛，不能据此声称连续解析几何边界已收敛。",
 "若数据/模型代表性审查未完成，不应自动将此报告的数值写入论文。"]
(ROOT/"RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n")
