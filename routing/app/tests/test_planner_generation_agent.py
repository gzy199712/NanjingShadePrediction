"""Regression checks for the planner-facing multimodal generation Agent contract."""

from pathlib import Path

from routing.app.backend import main


def test_upload_endpoint_merges_generation_agent_result(client, monkeypatch) -> None:
    semantic = {
        "public_walk_cycle_likelihood": "较高", "leaf_off_risk": False,
        "optimization_eligible": True, "panorama_detected": False,
        "width": 2048, "height": 512, "overlay_png_base64": "overlay",
    }
    agent = {
        "agent_name": "planner_panorama_generation_agent",
        "agent_state": "SAFE_ABSTENTION", "generation_available": False,
        "generation_blockers": ["tree_gap_consensus"], "tool_trace": [],
    }
    monkeypatch.setattr(main.PLANNER_ADAPTER, "analyze", lambda *args, **kwargs: semantic)
    monkeypatch.setattr(main.PLANNER_ADAPTER, "close", lambda: None)
    monkeypatch.setattr(main.PLANNER_GENERATION_AGENT, "analyze_and_generate", lambda *args, **kwargs: agent)
    response = client.post(
        "/api/planner/analyze-upload",
        files={"image": ("test.png", b"fake-image-bytes", "image/png")},
        data={"planner_intent": "连续乔木优先", "capture_month": "9"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_name"] == "planner_panorama_generation_agent"
    assert payload["agent_state"] == "SAFE_ABSTENTION"
    assert payload["generation_blockers"] == ["tree_gap_consensus"]
    assert payload["upload_persisted"] is False


def test_agent_pipeline_scripts_are_present() -> None:
    root = Path(__file__).resolve().parents[3]
    scripts = root / "streetscape/scripts"
    assert (scripts / "stage_46_powerpaint_gpu_editor_preflight.py").is_file()
    assert (scripts / "stage_46b_panorama_tree_generation_agent.py").is_file()
    assert (scripts / "stage_47_automatic_generation_acceptance.py").is_file()


def test_frontend_surfaces_agent_state_and_accepted_panorama() -> None:
    root = Path(__file__).resolve().parents[3]
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert "generated_panorama_png_base64" in script
    assert "generation_blockers" in script
    assert "多模态Agent" in script
    assert "renderAgentTrace" in script
    assert "renderPlannerImprovement" in script
    assert "/overlay" in script


def test_orchestrator_parses_intent_and_routes_tools() -> None:
    from routing.app.backend.planner_agent_orchestrator import build_agent_plan, parse_planner_intent
    intent = parse_planner_intent("沿道路南侧人行道形成连续高大乔木，不占机动车道")
    assert intent["preferred_intervention"] == "tree_first"
    assert {"道路南侧", "优先人行道", "保持机动车道", "连续树冠", "高大乔木"}.issubset(intent["constraints"])
    plan = build_agent_plan(intent, {
        "panorama_detected": True, "shade_optimization_need": "较高",
        "leaf_off_risk": False, "high_speed_road_suspected": False,
    })
    assert plan["selected_action"] == "directional_source_required"
    direction_plan = build_agent_plan(intent, {
        "panorama_detected": False, "shade_optimization_need": "较高",
        "leaf_off_risk": False, "high_speed_road_suspected": False,
    })
    assert direction_plan["selected_action"] == "directional_perspective_plan"
    assert "semantic_depth_25d_layout" in direction_plan["selected_tools"]
    assert plan["hidden_reasoning_exposed"] is False
    combined = parse_planner_intent("乔木种植走廊、树冠覆盖目标与人工遮阳补充复核区")
    assert combined["preferred_intervention"] == "tree_first"


def test_existing_point_overlay_does_not_invoke_generator(client, monkeypatch) -> None:
    point = client.get("/api/planner/points?hour=14").json()["points"][0]
    overlay = {
        "overlay_png_base64": "overlay", "overlay_legend": {"canopy_target": "树冠目标覆盖区域"},
        "suggestion_regions": [{"type": "canopy_target", "pixel_ratio": .08}],
        "space_type_assessment": "城市混合道路空间",
    }
    monkeypatch.setattr(main.PLANNER_ADAPTER, "analyze", lambda *args, **kwargs: overlay)
    monkeypatch.setattr(main.PLANNER_ADAPTER, "close", lambda: None)
    response = client.post(f"/api/planner/points/{point['point_id']}/overlay")
    assert response.status_code == 200
    payload = response.json()
    assert payload["generation_invoked"] is False
    assert payload["external_request_made"] is False


def test_overlay_confidence_reaches_gpu_worker_contract(client, monkeypatch) -> None:
    point = client.get("/api/planner/points?hour=14").json()["points"][0]
    captured = {}
    def fake_analyze(*args):
        captured["confidence"] = args[3]
        return {"overlay_png_base64": "overlay", "suggestion_regions": [], "public_space_score": .55}
    monkeypatch.setattr(main.PLANNER_ADAPTER, "analyze", fake_analyze)
    monkeypatch.setattr(main.PLANNER_ADAPTER, "close", lambda: None)
    response = client.post(f"/api/planner/points/{point['point_id']}/overlay?confidence=85")
    assert response.status_code == 200
    assert captured["confidence"] == 85


def test_frontend_has_confidence_and_public_space_controls() -> None:
    root = Path(__file__).resolve().parents[3]
    html = (root / "routing/app/frontend/templates/index.html").read_text(encoding="utf-8")
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert 'id="planner-confidence"' in html
    assert 'id="public-space-meter"' in html
    assert "optimization_azimuth_lines" in script
    assert "public_space_score" in script


def test_planner_overlay_and_map_are_visually_reduced() -> None:
    root = Path(__file__).resolve().parents[3]
    html = (root / "routing/app/frontend/templates/index.html").read_text(encoding="utf-8")
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    worker = (root / "routing/app/backend/planner_analysis_worker.py").read_text(encoding="utf-8")
    assert 'id="planner-map-layer"' not in html
    assert "遮荫优化方案类别" in html
    assert "canopy_height_curve" not in script and "canopy_height_curve" not in worker
    assert "_road_direction_exclusion" in worker and "_canopy_convex_envelope" in worker
    assert 'const meta={canopy_target:' in script
    for removed in ('"existing_effective_shade"', '"tree_corridor"', '"walk_demand"'):
        assert removed not in worker


def test_directional_planner_contract_is_visible() -> None:
    root = Path(__file__).resolve().parents[3]
    html = (root / "routing/app/frontend/templates/index.html").read_text(encoding="utf-8")
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    main_source = (root / "routing/app/backend/main.py").read_text(encoding="utf-8")
    assert "上传单一方位街景图像" in html
    assert 'id="planner-direction-strip"' in html
    assert "loadPlannerDirectionFusion" in script and "analyzePlannerDirection" in script
    assert "/directions/{heading}/analyze" in main_source
    assert "/directions/analyze-all" in main_source
    assert "nanjing_dinov2_morphology_transfer_reference" in main_source


def test_directional_planner_uses_curb_scale_and_shared_seam_geometry() -> None:
    root = Path(__file__).resolve().parents[3]
    worker = (root / "routing/app/backend/planner_analysis_worker.py").read_text(encoding="utf-8")
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert "_road_scene_geometry" in worker
    assert "curb_perspective_strength" in worker
    assert "road_to_ego_hood_width_ratio" in worker
    assert "same_capture_vehicle_calibrated_width" in worker
    assert "centre-dominant feather weights" in worker
    assert "sampled_rgb" in worker and "sampled_canopy" in worker
    assert "道路/采样车宽比" in script


def test_directional_planner_uses_depth_cross_section_and_facility_anchors() -> None:
    root = Path(__file__).resolve().parents[3]
    worker = (root / "routing/app/backend/planner_analysis_worker.py").read_text(encoding="utf-8")
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert "Depth Anything V2 Metric Outdoor Small" in worker
    assert "_depth_sidewalk_facility_geometry" in worker
    assert "depth_recovered_walk_corridor_ratio" in worker
    assert "distant_shade_does_not_cover_near_corridor" in worker
    assert "attached_artificial_shade_feasible" in worker
    assert "facility_anchor" in worker and "facility_anchor" in script
    assert "道路/采样车头宽度比" in script
    assert "fragmented_local_width_measurement" in worker
    assert "spatial_anchor_available" in worker


def test_directional_planner_uses_osm_urban_edges_and_local_shade_gaps() -> None:
    root = Path(__file__).resolve().parents[3]
    main = (root / "routing/app/backend/main.py").read_text(encoding="utf-8")
    worker = (root / "routing/app/backend/planner_analysis_worker.py").read_text(encoding="utf-8")
    script = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    assert "planner_osm_road_contexts" in main
    assert "formal_osm_point_edge_mapping" in main
    assert "osm_motor_vehicle_only_strong_gate" in worker
    assert "near_continuous_building_tree_street_edge" in worker
    assert "person_or_two_wheeler_detected" in worker
    assert "directional_shade_gap_detected" in worker
    assert "local_gap_headings" in worker
    assert "OSM道路先验" in script and "道路两侧连续性" in script
