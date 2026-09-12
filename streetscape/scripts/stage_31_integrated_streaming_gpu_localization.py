"""Integrated, resumable CUDA localization with frozen multiview tree evidence."""
from __future__ import annotations

import argparse,gc,json,logging,math,sqlite3,time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation,AutoModelForZeroShotObjectDetection,AutoProcessor,Mask2FormerForUniversalSegmentation
import yaml

from stage_29_streaming_gpu_localization import (database,export_results,group_matrix,object_metrics,object_prompt,
    raw_image,resolve_jobs,semantic_depth_metrics,upsert_base)

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'streetscape/configs/stage_31_integrated_streaming_pilot.yaml'

def args():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=DEFAULT);p.add_argument('--mode',required=True,choices=('check','run'))
 p.add_argument('--max-points',type=int);p.add_argument('--overwrite',action='store_true');p.add_argument('--stop-after-views',type=int);return p.parse_args()

def merge(base,override):
 out=dict(base)
 for key,value in override.items():out[key]=merge(out.get(key,{}),value) if isinstance(value,dict) and isinstance(out.get(key),dict) else value
 return out

def config(path):
 override=yaml.safe_load(path.read_text(encoding='utf-8'));base=yaml.safe_load(Path(override.pop('base_config')).read_text(encoding='utf-8'));return merge(base,override)

def logger(path):
 path.parent.mkdir(parents=True,exist_ok=True);x=logging.getLogger('stage_31');x.handlers.clear();x.setLevel(logging.INFO);f=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
 for h in (logging.FileHandler(path,encoding='utf-8'),logging.StreamHandler()):h.setFormatter(f);x.addHandler(h)
 return x

def delta(a,b):return abs((a-b+180)%360-180)
def viou(a1,a2,b1,b2):
 overlap=max(0.,min(a2,b2)-max(a1,b1));union=max(a2,b2)-min(a1,b1);return overlap/union if union else 0.

def finalize(frame,cfg,out,connection):
 rows=[]
 for view in frame.itertuples(index=False):
  for index,box in enumerate(json.loads(view.tree_boxes_json if isinstance(view.tree_boxes_json,str) else '[]')):
   center=(float(view.heading)+((box['x1']+box['x2'])/2-.5)*90)%360;half=max(.25,(box['x2']-box['x1'])*45)
   rows.append({'point_id':str(view.point_id),'heading':int(view.heading),'month':int(view.month),'box_index':index,**box,'world_azimuth_center':center,'world_half_span':half})
 evidence=pd.DataFrame(rows);repeats=[False]*len(evidence)
 if len(evidence):
  for _,indices in evidence.groupby(['point_id','class_name']).groups.items():
   ids=list(indices)
   for i in ids:
    a=evidence.loc[i]
    for j in ids:
     if i==j:continue
     b=evidence.loc[j]
     if delta(float(a.heading),float(b.heading))<=cfg['tree_filter']['adjacent_heading_maximum_degrees'] and delta(a.world_azimuth_center,b.world_azimuth_center)<=a.world_half_span+b.world_half_span+cfg['tree_filter']['adjacent_world_overlap_margin_degrees'] and viou(a.y1,a.y2,b.y1,b.y2)>=cfg['tree_filter']['vertical_iou_minimum']:
      repeats[i]=True;break
  evidence['adjacent_view_repeat']=repeats;evidence['semantic_vegetation_pass']=evidence.vegetation_support>=cfg['tree_filter']['vegetation_support_minimum']
  evidence['hard_leaf_off']=evidence.month.isin(cfg['tree_filter']['hard_leaf_off_months']);evidence['phenology_uncertain']=evidence.month.isin(cfg['tree_filter']['phenology_uncertain_months'])
  evidence['season_aware_tree_evidence']=evidence.adjacent_view_repeat&(evidence.semantic_vegetation_pass|evidence.hard_leaf_off)
 evidence.to_csv(out/'tree_detection_evidence.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
 points=[]
 for pid,g in frame.groupby('point_id'):
  e=evidence[evidence.point_id==pid] if len(evidence) else evidence;month=int(g.month.iloc[0]);count=int(e.season_aware_tree_evidence.sum()) if len(e) else 0
  points.append({'point_id':pid,'month':month,'hard_leaf_off':month in cfg['tree_filter']['hard_leaf_off_months'],'phenology_uncertain':month in cfg['tree_filter']['phenology_uncertain_months'],
   'raw_detection_count':len(e),'adjacent_repeat_count':int(e.adjacent_view_repeat.sum()) if len(e) else 0,'semantic_supported_count':int(e.semantic_vegetation_pass.sum()) if len(e) else 0,
   'season_aware_evidence_count':count,'multiview_trunk_count':int(((e.class_name=='tree_trunk')&e.season_aware_tree_evidence).sum()) if len(e) else 0,
   'decision':'EXISTING_TREE_EVIDENCE_PRESENT' if count else 'TREE_EVIDENCE_INSUFFICIENT'})
 point=pd.DataFrame(points);point.to_csv(out/'point_season_aware_tree_evidence.csv',index=False,encoding='utf-8-sig');point.to_sql('point_evidence',connection,if_exists='replace',index=False);connection.commit()
 conflicts={'month_only_evidence':int((evidence.hard_leaf_off&~evidence.adjacent_view_repeat&evidence.season_aware_tree_evidence).sum()) if len(evidence) else 0,
  'leaf_on_without_semantic_support':int((~evidence.hard_leaf_off&~evidence.semantic_vegetation_pass&evidence.season_aware_tree_evidence).sum()) if len(evidence) else 0,'artificial_shade_automatic':0}
 return evidence,point,conflicts

def write_summary(out,payload):
 (out/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(payload,ensure_ascii=False,indent=2))

def main():
 a=args();cfg=config(a.config);manifest_path=Path(cfg['inputs']['manifest']);raw=Path(cfg['inputs']['raw_direction_directory']);manifest=pd.read_csv(manifest_path,dtype={'point_id':str}) if manifest_path.exists() else pd.DataFrame()
 limit=a.max_points or int(cfg['processing']['pilot_point_count']);jobs=resolve_jobs(manifest,raw,limit) if len(manifest) else pd.DataFrame();models={k:Path(v['local_path']) for k,v in cfg['models'].items()}
 check={'status':'READY' if torch.cuda.is_available() and len(jobs)==limit*12 and jobs.filepath.ne('').all() and all(p.exists() for p in models.values()) else 'BLOCKED','point_count':limit,'view_count':len(jobs),'missing_images':int(jobs.filepath.eq('').sum()) if len(jobs) else None,
  'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cpu_fallback':False,'full_tensor_persistence':False,'integrated_tree_filter':True}
 if a.mode=='check':print(json.dumps(check,ensure_ascii=False,indent=2));return 0 if check['status']=='READY' else 1
 if check['status']!='READY':raise RuntimeError(check)
 out=Path(cfg['outputs']['directory']);out.mkdir(parents=True,exist_ok=True);db=database(out/'checkpoint.sqlite',a.overwrite);upsert_base(db,jobs);log=logger(Path(cfg['outputs']['log']));device=torch.device(cfg['gpu_policy']['device']);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
 start=datetime.now().astimezone();tick=time.perf_counter();fail=[];processed=0;log.info('START points=%d views=%d stop_after=%s',limit,len(jobs),a.stop_after_views)
 processor=AutoProcessor.from_pretrained(models['objects'],local_files_only=True);obj=AutoModelForZeroShotObjectDetection.from_pretrained(models['objects'],local_files_only=True,dtype=torch.float16).to(device).eval()
 if any(p.device.type!='cuda' for p in obj.parameters()):raise RuntimeError('non-CUDA object model')
 text,masks=object_prompt(processor,cfg['objects']['taxonomy'],device);pending=pd.read_sql_query('SELECT point_id,heading,filename FROM view_results WHERE object_done=0 ORDER BY CAST(point_id AS INTEGER),heading',db)
 for row in tqdm(pending.itertuples(index=False),total=len(pending),desc='stage_31 objects',unit='view',dynamic_ncols=True):
  try:
   _,rgb=raw_image(Path(row.filename),device);m=object_metrics(obj,rgb,text,masks,cfg);cols=list(m);db.execute(f"UPDATE view_results SET object_done=1,{','.join(x+'=?' for x in cols)},error_message=NULL,updated_at=? WHERE point_id=? AND heading=?",[*m.values(),datetime.now().astimezone().isoformat(),row.point_id,int(row.heading)]);db.commit();processed+=1
  except Exception as e:fail.append({'point_id':row.point_id,'heading':row.heading,'stage':'objects','error_message':f'{type(e).__name__}: {e}'})
  if a.stop_after_views and processed>=a.stop_after_views:
   frame=export_results(db,out);summary={**check,'status':'CONTROLLED_INTERRUPTION_COMPLETE','processed_this_run':processed,'object_done_count':int(frame.object_done.sum()),'semantic_depth_done_count':int(frame.semantic_depth_done.sum()),'failure_count':len(fail),'resumable':True,'formal_run_started':False};write_summary(out,summary);db.close();return 0
 del obj,processor,text,masks;gc.collect();torch.cuda.empty_cache()
 sem=Mask2FormerForUniversalSegmentation.from_pretrained(models['semantic'],local_files_only=True,dtype=torch.float16).to(device).eval();dep=AutoModelForDepthEstimation.from_pretrained(models['depth'],local_files_only=True,dtype=torch.float16).to(device).eval()
 if any(p.device.type!='cuda' for model in (sem,dep) for p in model.parameters()):raise RuntimeError('non-CUDA semantic/depth model')
 labels={int(k):v for k,v in sem.config.id2label.items()};groups=group_matrix(labels,cfg['semantic_groups'],device);pending=pd.read_sql_query('SELECT point_id,heading,filename,tree_boxes_json FROM view_results WHERE semantic_depth_done=0 ORDER BY CAST(point_id AS INTEGER),heading',db)
 for row in tqdm(pending.itertuples(index=False),total=len(pending),desc='stage_31 semantic+depth',unit='view',dynamic_ncols=True):
  try:
   _,rgb=raw_image(Path(row.filename),device);m=semantic_depth_metrics(sem,dep,rgb,groups,cfg,row.tree_boxes_json);cols=list(m);db.execute(f"UPDATE view_results SET semantic_depth_done=1,{','.join(x+'=?' for x in cols)},error_message=NULL,updated_at=? WHERE point_id=? AND heading=?",[*m.values(),datetime.now().astimezone().isoformat(),row.point_id,int(row.heading)]);db.commit();processed+=1
  except Exception as e:fail.append({'point_id':row.point_id,'heading':row.heading,'stage':'semantic_depth','error_message':f'{type(e).__name__}: {e}'})
 del sem,dep,groups;gc.collect();torch.cuda.empty_cache();frame=export_results(db,out);evidence,points,conflicts=finalize(frame,cfg,out,db);pd.DataFrame(fail,columns=('point_id','heading','stage','error_message')).to_csv(out/'failed_files.csv',index=False,encoding='utf-8-sig')
 complete=int(((frame.object_done==1)&(frame.semantic_depth_done==1)).sum());status='INTEGRATED_STREAMING_COMPLETE' if complete==len(jobs) and not fail and not any(conflicts.values()) else 'INTEGRATED_STREAMING_INCOMPLETE'
 summary={**check,'status':status,'started_at':start.isoformat(),'ended_at':datetime.now().astimezone().isoformat(),'runtime_seconds':round(time.perf_counter()-tick,3),'processed_this_run':processed,'success_view_count':complete,'failure_count':len(fail),
  'raw_tree_box_count':len(evidence),'filtered_tree_evidence_count':int(evidence.season_aware_tree_evidence.sum()) if len(evidence) else 0,'point_decision_counts':points.decision.value_counts().to_dict(),'conflicts':conflicts,
  'checkpoint_bytes':(out/'checkpoint.sqlite').stat().st_size,'full_tensor_files_written':0,'all_models_cuda':True,'peak_reserved_mib':round(torch.cuda.max_memory_reserved(device)/2**20,2),'resumable':True,'formal_run_started':limit==6014}
 write_summary(out,summary);log.info('END %s',summary);db.close();return 0 if status.endswith('COMPLETE') else 1

if __name__=='__main__':raise SystemExit(main())
