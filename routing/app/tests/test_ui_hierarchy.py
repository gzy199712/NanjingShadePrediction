"""Static regression checks for the navigation-first web interface."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HTML = (ROOT / "routing/app/frontend/templates/index.html").read_text(encoding="utf-8")
CSS = (ROOT / "routing/app/frontend/static/style.css").read_text(encoding="utf-8")
JS = (ROOT / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")


def test_primary_information_hierarchy() -> None:
    assert "选择更舒适的出行路线" in HTML
    assert '<button class="primary" id="compare-routes">比较三类路线</button>' in HTML
    assert 'id="route-metrics-details" hidden' in HTML
    assert "用一句话设置行程" in HTML and '<details class="assistant-panel">' in HTML


def test_accessible_navigation_controls() -> None:
    assert 'class="skip-link"' in HTML
    assert 'id="route-map" tabindex="0"' in HTML
    assert 'id="planner-point-map" tabindex="0"' in HTML
    assert "handleMapKey" in JS and "handlePlannerMapKey" in JS
    assert "prefers-reduced-motion" in CSS


def test_three_route_public_interface_only() -> None:
    assert "risk_aware" not in HTML
    assert "比较三类路线" in HTML
    assert "route-key-metrics" in JS


def test_responsive_navigation_design_tokens() -> None:
    assert "--surface-soft:" in CSS
    assert "@media (max-width: 900px)" in CSS
    assert "@media (max-width: 620px)" in CSS
    assert "min-height: 44px" in CSS


def test_planner_svf_phenology_and_building_lod_controls() -> None:
    assert 'id="planner-capture-month"' in HTML
    assert 'id="planner-show-buildings" type="checkbox" checked' in HTML
    assert 'class="grade-seasonal"' in HTML
    assert "plannerDecisionPipeline" in JS
    assert "schedulePlannerVisualRefresh" in JS
    assert ".planner-stage-flow" in CSS
    assert ".grade-seasonal" in CSS


def test_planner_dual_map_and_dual_scenario_outputs() -> None:
    assert 'id="planner-map-layer-plan"' in HTML
    assert 'id="planner-map-layer-shade"' in HTML
    assert 'class="planner-scenario-comparison"' in HTML
    assert 'id="planner-main-image"' in HTML
    assert 'id="planner-realistic-image"' in HTML
    assert 'id="generate-realistic-tree"' in HTML
    assert "setPlannerMapLayer" in JS and "SHADE_META" in JS
    assert "generateRealisticTreeScenario" in JS
    assert "realistic_tree_png_base64" in JS
    assert ".planner-segmented" in CSS and ".planner-scenario-comparison" in CSS


def test_planner_mask_editor_and_progressive_disclosure() -> None:
    assert 'class="planner-quick-flow"' in HTML
    assert 'id="planner-mask-canvas"' in HTML
    assert 'id="planner-brush-add"' in HTML and 'id="planner-brush-erase"' in HTML
    assert 'id="planner-feedback-consent"' in HTML
    assert 'id="planner-feedback-consent" checked' in HTML
    assert 'id="planner-no-intervention"' in HTML
    assert 'id="planner-no-intervention-submit"' in HTML
    assert 'id="planner-mask-submit" disabled' in HTML
    assert 'id="planner-feedback-status"' in HTML
    assert 'id="planner-feedback-proof"' in HTML
    assert 'id="planner-brush-cursor"' in HTML
    assert "initializePlannerMaskEditor" in JS
    assert "plannerManualMaskPayload" in JS
    assert "submitPlannerMask" in JS
    assert "updatePlannerBrushCursor" in JS
    assert ".planner-mask-editor" in CSS
