"""Regression checks for planner SVF-first screening and phenology gates."""


def test_planner_points_use_svf_primary_and_account_for_all_points(client) -> None:
    response = client.get("/api/planner/points?hour=14")
    assert response.status_code == 200
    result = response.json()
    assert result["classification_method"].startswith("svf_primary")
    assert result["thresholds"]["svf"] == {
        "good_max": 0.15, "fair_max": 0.25, "poor_max": 0.35
    }
    assert sum(result["grade_counts"].values()) == result["point_count"] == 8975
    # Only Nov/Dec/Jan/Feb are hard leaf-off months. March remains visible with
    # an uncertainty flag instead of being removed from the planning cascade.
    assert sum(point.get("march_phenology_uncertain", False) for point in result["points"]) > 2000


def test_low_svf_points_do_not_trigger_new_shade_intervention(client) -> None:
    points = client.get("/api/planner/points?hour=14").json()["points"]
    candidates = [
        point for point in points
        if point["svf"] <= 0.15
        and not point["leaf_off_risk"]
        and not point["scene_review_required"]
        and point["planner_citywide_class"] == "NO_INTERVENTION_LOW_SVF"
    ]
    assert candidates
    assert all(point["grade"] == "good" for point in candidates)
    assert all(not point["optimization_eligible"] for point in candidates)
    assert all(point["decision_stage"] == "stage_2_svf_need_gate" for point in candidates)


def test_citywide_final_decisions_replace_pending_gpu_state(client) -> None:
    points = client.get("/api/planner/points?hour=14").json()["points"]
    counts: dict[str, int] = {}
    for point in points:
        key = point["planner_citywide_class"]
        counts[key] = counts.get(key, 0) + 1
    assert counts["MOTOR_VEHICLE_ONLY_EXCLUDED"] == 771
    assert "PENDING_GPU_VISUAL_LOCALIZATION" not in counts
    assert counts["EXISTING_TREE_ROW_COVERAGE_REVIEW"] == 3198
    assert counts["EXISTING_TREE_PHENOLOGY_REVIEW"] == 1464
    assert counts["EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW"] == 561
    assert counts["TREE_PRIORITY_CANDIDATE"] == 4
    assert all(
        point["gpu_localization_complete"]
        for point in points
        if point["planner_citywide_class"] in {
            "EXISTING_TREE_ROW_COVERAGE_REVIEW",
            "EXISTING_TREE_PHENOLOGY_REVIEW",
            "EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW",
            "TREE_PRIORITY_CANDIDATE",
        }
    )
    # The frozen decision is preserved for reproducibility, while the live
    # planner may also create maturation/gap-completion scenarios for existing
    # tree rows that still have an SVF deficit. Road-centre capture is not
    # allowed to terminate 777 scenes solely because pavement pixels are few.
    assert counts["POTENTIAL_ACTIVE_MOBILITY_SHADE_REVIEW"] == 777
    assert sum(point["frozen_optimization_eligible"] for point in points) == 5
    assert sum(point["interactive_scenario_eligible"] for point in points) == 4541
    assert sum(point["optimization_eligible"] for point in points) == 4541
    assert all(
        point["scenario_generation_mode"] == "existing_tree_maturation_or_gap_completion"
        for point in points
        if point["interactive_scenario_eligible"]
        and point["planner_citywide_class"] in {
            "EXISTING_TREE_ROW_COVERAGE_REVIEW",
            "EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW",
        }
    )
    rescued = [point for point in points if point["planner_citywide_class"] == "POTENTIAL_ACTIVE_MOBILITY_SHADE_REVIEW"]
    assert all(point["space_type_is_probabilistic"] for point in rescued)
    assert all(point["space_type_assessment"] == "城市混合道路空间" for point in rescued)
    assert all("optimization_class" in point and "optimization_metrics" in point for point in points)
    assert all(0 <= point["public_space_score"] <= 1 for point in points)


def test_optional_manual_review_file_can_be_absent(client) -> None:
    response = client.get("/api/planner/points?hour=14")
    assert response.status_code == 200
    assert response.json()["point_count"] == 8975
