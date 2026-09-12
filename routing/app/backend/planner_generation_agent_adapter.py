"""Sequential, memory-safe adapter for the planner-facing panorama generation agent."""

from __future__ import annotations
import base64,csv,json,shutil,subprocess,uuid
from pathlib import Path
from typing import Any
import pandas as pd
from routing.app.backend.planner_agent_orchestrator import (
    build_agent_plan, parse_planner_intent, replan_record,
)

ROOT=Path(__file__).resolve().parents[3]
PYTHON=Path(r"D:\miniconda3\envs\gptthermalcomfort\python.exe")
POWERPAINT_PYTHON=Path(r"D:\miniconda3\envs\gptthermalcomfort-powerpaint\python.exe")


class PlannerGenerationAgentAdapter:
    def _run(self,command:list[str],cwd:Path,log:Path,timeout:int=900)->None:
        completed=subprocess.run(command,cwd=str(cwd),capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));log.write_text(completed.stdout+"\n--- STDERR ---\n"+completed.stderr,encoding="utf-8")
        if completed.returncode!=0:raise RuntimeError(f"Agent tool failed ({completed.returncode}): {Path(command[1]).name}; see {log.name}")

    def analyze_and_generate(self,image_path:Path,planner_intent:str,capture_month:int|None,semantic_result:dict[str,Any])->dict[str,Any]:
        task=ROOT/"routing/app/runtime_uploads"/f"agent_{uuid.uuid4().hex}";task.mkdir(parents=True,exist_ok=True);trace=[]
        try:
            intent_contract=parse_planner_intent(planner_intent);agent_plan=build_agent_plan(intent_contract,semantic_result)
            trace.append({"phase":"observe","tool":"mask2former_semantic_overlay","status":"PASS","evidence":{"svf":semantic_result.get("svf"),"space_type":semantic_result.get("space_type_assessment"),"shade_need":semantic_result.get("shade_optimization_need")}})
            trace.append({"phase":"plan","tool":"bounded_agent_controller","status":"PASS","evidence":{"selected_action":agent_plan["selected_action"],"selected_tools":agent_plan["selected_tools"]}})
            agent_image=image_path
            if bool(semantic_result.get("panorama_detected")):
                agent_image=task/"normalized_panorama.png";self._run([str(PYTHON),str(ROOT/"routing/app/backend/planner_panorama_normalize_worker.py"),"--input",str(image_path),"--output",str(agent_image)],ROOT,task/"00_normalize.log");trace.append({"phase":"act","tool":"cuda_panorama_normalization","status":"PASS","evidence":{"width":2048,"height":512}})
            vlm_path=task/"vlm.json";self._run([str(PYTHON),str(ROOT/"routing/app/backend/planner_vlm_agent_worker.py"),"--image",str(agent_image),"--output",str(vlm_path),"--intent",planner_intent],ROOT,task/"01_vlm.log");vlm=json.loads(vlm_path.read_text(encoding="utf-8"));trace.append({"phase":"observe","tool":"qwen3_vl_consensus","status":"PASS","evidence":{"structural_scene":vlm["structural_scene"],"tree_state":vlm["tree_state"],"planning_action":vlm.get("planning_action")}})
            gates={"space_type_not_motor_only":not bool(semantic_result.get("high_speed_road_suspected")),"non_winter_phenology":not bool(semantic_result.get("leaf_off_risk")),"svf_need":bool(semantic_result.get("optimization_eligible")),"confidence_for_formal_generation":int(semantic_result.get("planning_confidence",65))>=50,"panorama_geometry":bool(semantic_result.get("panorama_detected")),"ground_street_consensus":vlm["structural_scene"]=="GROUND_STREET","tree_gap_consensus":vlm["tree_state"]=="TREE_GAP_WITH_PLANTING_SPACE"}
            blockers=[name for name,value in gates.items() if not value];trace.append({"tool":"deterministic_pre_generation_controller","status":"PASS","evidence":gates})
            base={"agent_name":"planner_multimodal_tool_agent","agent_operational":True,"agent_plan":agent_plan,"intent_contract":intent_contract,"vlm":vlm,"generation_gates":gates,"generation_blockers":blockers,"tool_trace":trace,"planner_intent_applied":planner_intent,"capture_month":capture_month,"generation_available":False,"automatic_acceptance":False}
            if agent_plan["selected_action"]!="tree_generation_with_acceptance":
                state="DIRECTIONAL_PLAN_COMPLETE" if agent_plan["selected_action"]=="directional_perspective_plan" else "PLANNING_COMPLETE_WITH_EVIDENCE_OVERLAY"
                explanation=" 已完成道路尽头排除、左右侧透视布局、自监督案例参照与遮荫方案叠加。" if state=="DIRECTIONAL_PLAN_COMPLETE" else " 已完成遮荫证据叠加与空间类型判断，未调用不适合的图像生成器。"
                return {**base,"agent_state":state,"agent_explanation":agent_plan["decision_summary"]+explanation}
            if blockers:
                trace.append(replan_record("tree_generation_with_acceptance",",".join(blockers),"evidence_overlay_and_planning_report",1))
                return {**base,"tool_trace":trace,"agent_state":"REPLANNED_TO_SAFE_EVIDENCE_OVERLAY","agent_explanation":"生成门槛未全部通过；Agent已自动改用半透明规划叠加和可审计建议，而不是终止整个分析。"}
            point_id="upload";agent_csv=task/"agent.csv";benchmark_csv=task/"benchmark.csv"
            with agent_csv.open("w",newline="",encoding="utf-8-sig") as handle:writer=csv.DictWriter(handle,fieldnames=["point_id","policy_state","next_tool"]);writer.writeheader();writer.writerow({"point_id":point_id,"policy_state":"READY_FOR_LAYOUT","next_tool":"propose_tree_layout"})
            with benchmark_csv.open("w",newline="",encoding="utf-8-sig") as handle:writer=csv.DictWriter(handle,fieldnames=["point_id","benchmark_role","output_path","width","height","north_at_x0","panorama_exists"]);writer.writeheader();writer.writerow({"point_id":point_id,"benchmark_role":"GROUND_WALKABLE_TREE_GAP", "output_path":str(agent_image.resolve()),"width":2048,"height":512,"north_at_x0":True,"panorama_exists":True})
            windows=task/"windows";layout=task/"layout";structure=task/"structure";generation=task/"generation";acceptance=task/"acceptance"
            self._run([str(PYTHON),str(ROOT/"streetscape/scripts/stage_44_panorama_perspective_windows.py"),"--mode","run","--overwrite","--agent-results",str(agent_csv),"--benchmark",str(benchmark_csv),"--output-directory",str(windows),"--report",str(task/"step44.md")],ROOT,task/"02_windows.log");trace.append({"tool":"gpu_spherical_windows","status":"PASS"})
            self._run([str(PYTHON),str(ROOT/"streetscape/scripts/stage_45_tree_first_semantic_25d_layout.py"),"--mode","run","--overwrite","--agent-results",str(agent_csv),"--windows-directory",str(windows),"--output-directory",str(layout),"--report",str(task/"step45.md")],ROOT,task/"03_layout.log");layout_state=json.loads((layout/"points"/point_id/"layout.json").read_text(encoding="utf-8"));trace.append({"tool":"semantic_depth_25d_layout","status":layout_state["status"],"evidence":{"tree_anchor_count":len(layout_state["tree_anchors"]),"selected_heading":layout_state["selected_heading"]}})
            if layout_state["status"]!="LAYOUT_READY":
                trace.append(replan_record("tree_generation_with_acceptance","no_valid_25d_layout","evidence_overlay_and_facility_review",1));return {**base,"tool_trace":trace,"agent_state":"REPLANNED_AFTER_LAYOUT_REJECTION","agent_explanation":"2.5D布局没有找到可信树列位置；Agent已回退为遮荫缺口叠加，并仅把人工遮阳作为规划师复核思路。"}
            self._run([str(PYTHON),str(ROOT/"streetscape/scripts/stage_45b_multimodal_structural_scene_gate.py"),"--mode","run","--overwrite","--layout-directory",str(layout),"--windows-directory",str(windows),"--benchmark",str(benchmark_csv),"--output-directory",str(structure),"--report",str(task/"step45b.md")],ROOT,task/"04_structure.log");sqc=pd.read_csv(structure/"structural_scene_qc.csv");support=bool(sqc.iloc[0].planting_support_pass);trace.append({"tool":"second_structural_scene_gate","status":"PASS" if support else "REJECT","evidence":{"structural_scene":str(sqc.iloc[0].structural_scene)}})
            if not support:
                trace.append(replan_record("tree_generation_with_acceptance","structural_scene_rejected","evidence_overlay_and_facility_review",1));return {**base,"tool_trace":trace,"agent_state":"REPLANNED_AFTER_STRUCTURAL_REJECTION","agent_explanation":"局部结构复核不支持直接种树；Agent保留缺口定位并转为人工遮阳补充复核。"}
            # A second independent full-panorama load must reproduce both VLM
            # decisions. This catches deterministic-decoding instability caused
            # by ambiguous early-spring trees or infrastructure-like scenes.
            vlm_repeat_path=task/"vlm_repeat.json";self._run([str(PYTHON),str(ROOT/"routing/app/backend/planner_vlm_agent_worker.py"),"--image",str(agent_image),"--output",str(vlm_repeat_path)],ROOT,task/"04b_vlm_repeat.log");vlm_repeat=json.loads(vlm_repeat_path.read_text(encoding="utf-8"));repeat_stable=vlm_repeat["structural_scene"]==vlm["structural_scene"]=="GROUND_STREET" and vlm_repeat["tree_state"]==vlm["tree_state"]=="TREE_GAP_WITH_PLANTING_SPACE";trace.append({"tool":"independent_vlm_repeatability_gate","status":"PASS" if repeat_stable else "REJECT","evidence":{"first_structural":vlm["structural_scene"],"repeat_structural":vlm_repeat["structural_scene"],"first_tree_state":vlm["tree_state"],"repeat_tree_state":vlm_repeat["tree_state"]}})
            if not repeat_stable:
                trace.append(replan_record("tree_generation_with_acceptance","vlm_not_repeatable","evidence_overlay_only",1));return {**base,"tool_trace":trace,"agent_state":"REPLANNED_AFTER_VLM_DISAGREEMENT","agent_explanation":"独立视觉判断不一致；Agent不发布生成图，但仍输出遮荫评估、空间类型和半透明规划位置。"}
            self._run([str(POWERPAINT_PYTHON),str(ROOT/"streetscape/scripts/stage_46b_panorama_tree_generation_agent.py"),"--mode","run","--overwrite","--layout-directory",str(layout),"--structural-qc",str(structure/"structural_scene_qc.csv"),"--benchmark",str(benchmark_csv),"--windows-directory",str(windows),"--output-directory",str(generation),"--report",str(task/"step46b.md")],ROOT,task/"05_generation.log");trace.append({"tool":"powerpaint_full_gpu_editor","status":"PASS"})
            self._run([str(PYTHON),str(ROOT/"streetscape/scripts/stage_47_automatic_generation_acceptance.py"),"--mode","run","--overwrite","--agent-state",str(generation/"agent_state.json"),"--generation-results",str(generation/"generation_results.csv"),"--output-directory",str(acceptance),"--report",str(task/"step47.md")],ROOT,task/"06_acceptance.log");aqc=pd.read_csv(acceptance/"acceptance_results.csv");accepted=aqc.loc[aqc.automatic_acceptance.astype(bool)];trace.append({"tool":"automatic_semantic_geometry_acceptance","status":"PASS" if len(accepted) else "REJECT","evidence":{"accepted_variants":len(accepted),"generated_variants":len(aqc)}})
            if not len(accepted):
                trace.append(replan_record("tree_generation_with_acceptance","automatic_quality_rejection","evidence_overlay_only",2));return {**base,"tool_trace":trace,"agent_state":"REPLANNED_AFTER_GENERATION_REJECTION","agent_explanation":"生成变体未通过语义、结构或保真验收；Agent自动回退到可信的规划叠加图，不展示伪结果。"}
            best=accepted.sort_values(["svf_reduction","vegetation_ratio_increase"],ascending=False).iloc[0];payload=base64.b64encode(Path(best.panorama_path).read_bytes()).decode("ascii")
            return {**base,"tool_trace":trace,"generation_available":True,"automatic_acceptance":True,"agent_state":"ACCEPTED_GENERATION","agent_explanation":"已通过全自动多模态、2.5D、结构、生成和语义保真门槛。","generated_panorama_png_base64":payload,"selected_seed":int(best.seed),"scenario_metrics":{"baseline_svf":float(best.baseline_svf),"scenario_svf":float(best.scenario_svf),"svf_reduction":float(best.svf_reduction),"vegetation_ratio_increase":float(best.vegetation_ratio_increase),"tmrt_utci_state":str(best.thermal_evaluation_state)}}
        finally:
            shutil.rmtree(task,ignore_errors=True)
