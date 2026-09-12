"""Merge deterministic gates and streamed multimodal evidence into 8,975 final decisions."""
from __future__ import annotations
import argparse,json,logging,shutil,time
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml

ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'streetscape/configs/stage_32_citywide_final_planning_decisions.yaml'
def args():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--mode',required=True,choices=('check','run'));p.add_argument('--overwrite',action='store_true');return p.parse_args()
def logger(path):
 path.parent.mkdir(parents=True,exist_ok=True);x=logging.getLogger('stage_32');x.handlers.clear();x.setLevel(logging.INFO);f=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
 for h in (logging.FileHandler(path,encoding='utf-8'),logging.StreamHandler()):h.setFormatter(f);x.addHandler(h)
 return x
def adjacent(headings):
 values={int(x)%360 for x in headings};return any((x+30)%360 in values or (x-30)%360 in values for x in values)
def main():
 a=args();cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'));paths={k:Path(v) for k,v in cfg['inputs'].items()};missing=[str(p) for p in paths.values() if not p.exists()]
 cascade=pd.read_csv(paths['cascade'],dtype={'point_id':str}) if not missing else pd.DataFrame();views=pd.read_csv(paths['view_metrics'],dtype={'point_id':str}) if not missing else pd.DataFrame();trees=pd.read_csv(paths['point_tree_evidence'],dtype={'point_id':str}) if not missing else pd.DataFrame();metrics=pd.read_csv(paths['point_metrics'],dtype={'point_id':str}) if not missing else pd.DataFrame()
 gpu_ids=set(cascade.loc[cascade.high_resolution_gpu_required==True,'point_id']) if len(cascade) else set();check={'status':'READY' if not missing and len(cascade)==8975 and len(views)==72168 and len(trees)==len(metrics)==6014 and gpu_ids==set(trees.point_id)==set(metrics.point_id) else 'BLOCKED','cascade_points':len(cascade),'gpu_points':len(gpu_ids),'view_count':len(views),'tree_points':len(trees),'metric_points':len(metrics),'missing_inputs':missing,'manual_labels_used':False}
 if a.mode=='check':print(json.dumps(check,ensure_ascii=False,indent=2));return 0 if check['status']=='READY' else 1
 if check['status']!='READY':raise RuntimeError(check)
 out=Path(cfg['outputs']['directory']);
 if out.exists() and a.overwrite:shutil.rmtree(out)
 if out.exists() and any(out.iterdir()):raise RuntimeError('outputs exist; use --overwrite')
 out.mkdir(parents=True);log=logger(Path(cfg['outputs']['log']));started=datetime.now().astimezone();tick=time.perf_counter();r=cfg['rules']
 views['walkable_direction']=(views.lower_half_walkable_probability>=r['walkable_lower_half_probability_minimum'])|(views.retained_walkable_ratio>=r['walkable_retained_ratio_minimum'])
 direction_rows=[]
 for pid,g in tqdm(views.groupby('point_id'),total=views.point_id.nunique(),desc='stage_32 walkable directions',unit='point',dynamic_ncols=True):
  headings=g.loc[g.walkable_direction,'heading'].astype(int).tolist();direction_rows.append({'point_id':pid,'walkable_direction_count':len(headings),'adjacent_walkable_direction_pair':adjacent(headings),'visible_walkable_multiview':len(headings)>=r['minimum_walkable_directions'] and (adjacent(headings) if r['require_adjacent_walkable_directions'] else True)})
 directions=pd.DataFrame(direction_rows)
 detections=pd.read_csv(paths['tree_detection_evidence'],dtype={'point_id':str},usecols=['point_id','heading','season_aware_tree_evidence'])
 accepted=detections[detections.season_aware_tree_evidence==True].groupby(['point_id','heading']).size().rename('accepted').reset_index();tree_dirs=accepted.groupby('point_id').heading.nunique().rename('tree_evidence_direction_count').reset_index()
 high=trees.merge(metrics,on='point_id',validate='one_to_one').merge(directions,on='point_id',validate='one_to_one').merge(tree_dirs,on='point_id',how='left',validate='one_to_one');high.tree_evidence_direction_count=high.tree_evidence_direction_count.fillna(0).astype(int)
 highmap={row.point_id:row for row in high.itertuples(index=False)};rows=[]
 for source in tqdm(cascade.itertuples(index=False),total=len(cascade),desc='stage_32 final decisions',unit='point',dynamic_ncols=True):
  row=source._asdict();pid=str(source.point_id)
  if pid not in highmap:
   row.update({'final_decision_stage':source.decision_stage_v2,'final_decision_class':source.decision_class_v2,'final_decision_status':source.decision_status_v2,'final_decision_basis':source.decision_basis_v2,'optimization_eligible_final':source.decision_class_v2=='ACTIONABLE_VISUAL_ADVICE','planner_review_required_final':source.decision_class_v2 in {'AUTO_PLANNING_ABSTAIN_INSUFFICIENT_VISIBLE_WALKABLE'},'gpu_localization_complete':False,'existing_tree_evidence':None,'walkable_direction_count':None,'tree_evidence_direction_count':None,'walkable_tree_cooccurrence_proxy':None})
  else:
   e=highmap[pid];existing=e.decision=='EXISTING_TREE_EVIDENCE_PRESENT';visible=bool(e.visible_walkable_multiview);hard=bool(e.hard_leaf_off);march=bool(e.phenology_uncertain)
   if hard or march:
    cls='EXISTING_TREE_PHENOLOGY_REVIEW' if existing else 'AUTO_PLANNING_ABSTAIN_PHENOLOGY_UNCERTAIN';stage='stage_2_phenology_gate';eligible=False;review=True;status='落叶或物候不确定影像：报告现状树列证据，暂停自动新增树木。';basis=f'month={int(e.month)}; existing_tree_evidence={existing}; March仅作不确定性标记。'
   elif not visible:
    cls='AUTO_PLANNING_ABSTAIN_INSUFFICIENT_VISIBLE_WALKABLE';stage='stage_1_public_space_gate';eligible=False;review=True;status='跨视角可步行空间证据不足，自动规划弃权。';basis=f'walkable_directions={int(e.walkable_direction_count)}; adjacent_pair={bool(e.adjacent_walkable_direction_pair)}。'
   elif existing:
    sparse=float(source.GVI)<r['sparse_tree_gvi_maximum'] or float(e.vegetation_ratio)<r['sparse_tree_semantic_vegetation_maximum'];cls='EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW' if sparse else 'EXISTING_TREE_ROW_COVERAGE_REVIEW';stage='stage_4_tree_evidence';eligible=False;review=True
    status='已有树木证据但绿量偏低：优先评估幼树生长、养护与冠幅成熟，不自动新增树木。' if sparse else '已有连续多视角树木证据：复核树冠对人行空间的覆盖和缺口，不自动新增树木。';basis=f'tree_evidence_directions={int(e.tree_evidence_direction_count)}; walkable_directions={int(e.walkable_direction_count)}; GVI={float(source.GVI):.4f}; semantic_vegetation={float(e.vegetation_ratio):.4f}。'
   else:
    cls='TREE_PRIORITY_CANDIDATE';stage='stage_5_optimization';eligible=True;review=True;status='未获得可靠现状树木证据且可步行空间稳定可见：进入树木优先候选定位。';basis=f'tree_evidence_count=0; walkable_directions={int(e.walkable_direction_count)}; SVF={float(source.SVF):.4f}。'
   co=min(int(e.tree_evidence_direction_count),int(e.walkable_direction_count))/max(1,int(e.walkable_direction_count))
   row.update({'final_decision_stage':stage,'final_decision_class':cls,'final_decision_status':status,'final_decision_basis':basis,'optimization_eligible_final':eligible,'planner_review_required_final':review,'gpu_localization_complete':True,'existing_tree_evidence':existing,'walkable_direction_count':int(e.walkable_direction_count),'tree_evidence_direction_count':int(e.tree_evidence_direction_count),'walkable_tree_cooccurrence_proxy':round(co,4),'semantic_sky_ratio':float(e.sky_ratio),'semantic_vegetation_ratio':float(e.vegetation_ratio),'retained_walkable_ratio':float(e.retained_walkable_ratio),'season_aware_tree_evidence_count':int(e.season_aware_evidence_count),'multiview_trunk_count':int(e.multiview_trunk_count)})
  rows.append(row)
 final=pd.DataFrame(rows);final.to_csv(out/'citywide_final_planning_decisions.csv.gz',index=False,compression='gzip',encoding='utf-8-sig');web_columns=['point_id','month','march_phenology_uncertain','final_decision_class','final_decision_stage','optimization_eligible_final','planner_review_required_final','gpu_localization_complete','existing_tree_evidence','walkable_direction_count','tree_evidence_direction_count','walkable_tree_cooccurrence_proxy','semantic_sky_ratio','semantic_vegetation_ratio','retained_walkable_ratio','season_aware_tree_evidence_count','multiview_trunk_count','final_decision_status','final_decision_basis'];final.reindex(columns=web_columns).to_csv(out/'planner_web_decisions.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
 counts=final.final_decision_class.value_counts().to_dict();conflicts={'pending_gpu_remaining':int((final.final_decision_class=='PENDING_GPU_VISUAL_LOCALIZATION').sum()),'motor_only_eligible':int(((final.final_decision_class=='MOTOR_VEHICLE_ONLY_EXCLUDED')&final.optimization_eligible_final).sum()),'low_svf_eligible':int(((final.final_decision_class=='NO_INTERVENTION_LOW_SVF')&final.optimization_eligible_final).sum()),'artificial_shade_automatic':0,'duplicate_points':int(final.point_id.duplicated().sum())}
 summary={**check,'status':'CITYWIDE_FINAL_DECISIONS_COMPLETE' if len(final)==8975 and not any(conflicts.values()) else 'CITYWIDE_FINAL_DECISIONS_CONFLICT','started_at':started.isoformat(),'ended_at':datetime.now().astimezone().isoformat(),'runtime_seconds':round(time.perf_counter()-tick,3),'final_point_count':len(final),'decision_counts':counts,'optimization_eligible_count':int(final.optimization_eligible_final.sum()),'planner_review_required_count':int(final.planner_review_required_final.sum()),'gpu_localization_complete_count':int(final.gpu_localization_complete.sum()),'conflicts':conflicts,'tree_policy':r['tree_policy'],'artificial_shade_policy':r['artificial_shade_policy'],'claim_boundary':'Citywide automatic planning screening; tree evidence and walkable co-occurrence are multiview proxies, not independent tree counts, canopy-shadow causality, or construction design.'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');Path(cfg['outputs']['report']).write_text('\n'.join(['# stage_32 全市最终规划决策', '',f"状态：`{summary['status']}`。",'',f"- 点位：{len(final):,}",f"- 已完成高分辨率GPU定位：{summary['gpu_localization_complete_count']:,}",f"- 可进入树木优先候选定位：{summary['optimization_eligible_count']:,}",f"- 待GPU状态残留：{conflicts['pending_gpu_remaining']}",'', '## 最终类别']+[f"- `{k}`：{v:,}" for k,v in counts.items()]+['','树木证据、可步行方向与二者共现均为多视角代理指标，不代表独立树木棵数、树冠实际投影遮荫或施工可行性。人工遮阳仅供规划师补充复核。'])+'\n',encoding='utf-8');log.info('END %s',summary);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if summary['status'].endswith('COMPLETE') else 1
if __name__=='__main__':raise SystemExit(main())
