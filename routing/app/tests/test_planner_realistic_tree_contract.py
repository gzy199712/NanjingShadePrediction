"""Contract checks for the mask-constrained realistic tree scenario path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MAIN = (ROOT / "routing/app/backend/main.py").read_text(encoding="utf-8")
WORKER = (ROOT / "routing/app/backend/planner_realistic_tree_worker.py").read_text(encoding="utf-8")
PBE_WORKER = (ROOT / "routing/app/backend/planner_paint_by_example_worker.py").read_text(encoding="utf-8")
POWERPAINT_WORKER = (ROOT / "routing/app/backend/planner_powerpaint_worker.py").read_text(encoding="utf-8")
MASK_WORKER = (ROOT / "routing/app/backend/planner_manual_mask_worker.py").read_text(encoding="utf-8")


def test_existing_and_upload_endpoints_expose_realistic_generation() -> None:
    assert '/api/planner/points/{point_id}/directions/{heading}/generate-realistic' in MAIN
    assert 'generate_realistic: int = Form(0)' in MAIN
    assert "PLANNER_REALISTIC_GENERATION_LOCK" in MAIN


def test_generation_is_tree_only_reference_guided_and_full_gpu() -> None:
    assert "proposal_np = label > 127" in WORKER
    assert "--reference" in WORKER and "foliage_palette" in WORKER
    assert 'pipe.to("cuda:0")' in WORKER
    assert 'CPU fallback is forbidden' in WORKER
    assert "outside_mask_max_difference" in WORKER
    assert "most_obvious_crown" in WORKER
    assert "same_image_most_obvious_crown_first" in WORKER
    assert "nearby_coordinate_most_obvious_crown_fallback" in WORKER
    assert "powerpaint_original_scene_shape_guided_diffusion_inpaint" in WORKER
    assert "paint_by_example_original_scene_exemplar_guided_inpaint" in PBE_WORKER
    assert "image=source" in WORKER and "image=source" in PBE_WORKER
    assert "reference_only_not_pixel_prefill" in WORKER
    assert "prepare_edit_masks" in WORKER and "prepare_edit_masks" in PBE_WORKER
    assert "target_region_fill_pass" in MAIN
    assert "planner_powerpaint_worker.py" in MAIN
    assert 'pipe.to("cuda:0")' in POWERPAINT_WORKER
    assert '"controlnet_inpaint"' not in MAIN
    assert "semantic_vegetation" in WORKER
    assert "vegetation_mask_png_base64" in MAIN
    assert "current_image_first_then_nearest_coordinate_good_shade" in MAIN


def test_backend_automatically_accepts_or_safely_rejects() -> None:
    assert "REALISTIC_TREE_SCENARIO_ACCEPTED" in MAIN
    assert "REALISTIC_SCENARIO_REJECTED" in MAIN
    assert "vegetation_gain" in MAIN
    assert "directional_sky_view_change" in MAIN
    assert "building_hallucination_pass" in MAIN
    assert "tree_semantic_pass" in MAIN
    assert "foliage_gain_inside_mask" in WORKER
    assert "billboard" in WORKER
    assert "REALISTIC_SCENARIO_REVIEW_REQUIRED" in MAIN
    assert "planner_review_candidate" in MAIN
    assert "target_vegetation_fraction_after" in MAIN
    assert "target_vegetation_gain" in MAIN


def test_road_end_is_a_two_dimensional_protected_wedge() -> None:
    worker = (ROOT / "routing/app/backend/planner_analysis_worker.py").read_text(encoding="utf-8")
    assert "def _road_end_protected_wedge" in worker
    assert "lane_marking_confidence" in worker
    assert "semantic_curb_pair_plus_bright_lane_marking_convergence_protected_wedge" in worker
    assert "canopy_target &= ~road_end_protected" in worker
    assert "road_end_proposal_overlap_pixels" in worker
    assert "road_end_protected_mask_png_base64" in worker


def test_manual_mask_is_guarded_and_feedback_requires_explicit_submission() -> None:
    assert "generate-realistic-edited" in MAIN
    assert "_decode_planner_manual_mask" in MAIN
    assert "manual &= ~protected" in MASK_WORKER
    assert "feedback_consent" in MAIN
    assert "online_training_performed" in MAIN
    assert "opt_in_rlhf_preference_learning_candidate" in MAIN
    assert '/api/planner/feedback/{feedback_id}' in MAIN
    assert "feedback_verification_url" in MAIN


def test_no_intervention_reason_has_a_feedback_record() -> None:
    models = (ROOT / "routing/app/backend/api_models.py").read_text(encoding="utf-8")
    html = (ROOT / "routing/app/frontend/templates/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert "PlannerNoInterventionReviewRequest" in models
    assert "/review-no-intervention" in MAIN
    assert "/review-upload-no-intervention" in MAIN
    assert "no_intervention_natural_language_reason" in MAIN
    assert 'id="planner-feedback-consent" checked' in html
    assert 'id="planner-no-intervention-reason"' in html
    assert "submitPlannerNoIntervention" in javascript
    assert 'reviewSubmit.disabled=false' in javascript
    assert 'plannerNoInterventionReason' in javascript
    assert '/api/planner/points/{point_id}/review-no-intervention' in MAIN
    assert 'review_scope": "point_panorama"' in MAIN
    assert 'plannerReviewScope' in javascript


def test_long_planner_tasks_expose_accessible_progress_feedback() -> None:
    html = (ROOT / "routing/app/frontend/templates/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    css = (ROOT / "routing/app/frontend/static/style.css").read_text(encoding="utf-8")
    assert 'id="planner-task-progress"' in html
    assert 'aria-live="polite"' in html
    assert 'id="planner-task-progress-bar"' in html
    assert "beginPlannerProgress" in javascript
    assert "finishPlannerProgress" in javascript
    assert "failPlannerProgress" in javascript
    assert ".planner-task-progress" in css


def test_generation_need_is_decoupled_from_automatic_spatial_anchor() -> None:
    worker = (ROOT / "routing/app/backend/planner_analysis_worker.py").read_text(encoding="utf-8")
    javascript = (ROOT / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert '"generation_candidate_available"' in worker
    assert '"automatic_generation_candidate_available"' in worker
    assert '"planner_assisted_manual_region_required"' in worker
    assert "请先微调树冠区域" in javascript
    assert "generation_candidate_available" in javascript
    assert 'if optimization_eligible and not high_speed_road_suspected' in worker
