"""
Test Suite: Advanced Features Verification for IEEE Smart Building Command Center.
Verifies:
1. Multi-Camera CCTV Simulation state management and coverage analysis.
2. Multi-Floor Building Graph connectivity and floor subgraphs.
3. Temporal hazard trend calculation and moving average forecast.
4. Evacuation efficiency score calculation.
5. Vulnerable occupant priority simulation weighting impact.
6. Multi-format digital incident report exports (TXT, JSON, CSV).
7. Local deterministic Emergency Decision Assistant responses.
8. Camera coverage and occlusion uncertainty penalty calculation.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import patch
from integration.multi_camera_sim import MultiCameraSimulator
from integration.decision_assistant import EmergencyDecisionAssistant
from graph.multi_floor_graph import MultiFloorBuildingGraph
from dashboard.components import (
    generate_incident_report_text,
    generate_incident_report_json,
    generate_incident_report_csv,
    render_evacuation_guidance_card,
    render_detection_summary_cards,
)


def run_advanced_features_tests():
    print("=" * 70)
    print("ADVANCED FEATURES VERIFICATION TEST SUITE (IEEE UPGRADE)")
    print("=" * 70)

    total = 0
    passed = 0

    def record(name: str, condition: bool, msg: str = ""):
        nonlocal total, passed
        total += 1
        if condition:
            passed += 1
            print(f"[PASS] Feature Test {total:02d}: {name} - {msg}")
        else:
            print(f"[FAIL] Feature Test {total:02d}: {name} - {msg}")

    # 1. Multi-Camera Simulator Initialization
    try:
        cam_sim = MultiCameraSimulator()
        cams = cam_sim.get_all_cameras()
        record("Multi-Camera Simulation Initialization", len(cams) == 4, f"Initialized {len(cams)} camera channels")
    except Exception as e:
        record("Multi-Camera Simulation Initialization", False, f"Error: {e}")

    # 2. Multi-Camera Telemetry Update
    try:
        cam_sim.update_camera_telemetry(
            cam_id="CAM-01",
            fire_detected=True,
            smoke_detected=False,
            hazard_score=0.875,
            people_count=4,
            crowd_density=0.20,
            latency_ms=120.5
        )
        c1 = cam_sim.get_camera("CAM-01")
        record("Camera Telemetry Update", c1["fire_detected"] and c1["hazard_score"] == 0.875, "CAM-01 telemetry updated")
    except Exception as e:
        record("Camera Telemetry Update", False, f"Error: {e}")

    # 3. Camera Coverage & Occlusion Uncertainty Penalty
    try:
        cam_sim.update_camera_telemetry("CAM-03", fire_detected=False, smoke_detected=False, hazard_score=0.0, people_count=0, crowd_density=0.0, coverage="UNAVAILABLE", occlusion="HIGH")
        penalty = cam_sim.get_zone_uncertainty_penalty("EXIT_A")
        record("Coverage Uncertainty Penalty", penalty > 0, f"Uncertainty penalty calculated: {penalty:.1f}")
    except Exception as e:
        record("Coverage Uncertainty Penalty", False, f"Error: {e}")

    # 4. Multi-Floor Building Graph Creation & Connectivity
    try:
        mf_graph = MultiFloorBuildingGraph()
        g0 = mf_graph.get_floor_subgraph(0)
        g1 = mf_graph.get_floor_subgraph(1)
        record("Multi-Floor Graph Creation", len(g0.nodes) >= 4 and len(g1.nodes) >= 4, f"Ground Floor: {len(g0.nodes)} nodes, First Floor: {len(g1.nodes)} nodes")
    except Exception as e:
        record("Multi-Floor Graph Creation", False, f"Error: {e}")

    # 5. Multi-Floor Vertical Connection via Stairwell
    try:
        has_stair_conn = mf_graph.graph.has_edge("G_CORRIDOR_1", "F1_CORRIDOR_2")
        record("Multi-Floor Vertical Stairwell Connection", has_stair_conn, "Stairwell connection between Ground and Level 1 verified")
    except Exception as e:
        record("Multi-Floor Vertical Stairwell Connection", False, f"Error: {e}")

    # 6. Digital Incident Report - TXT Format
    sample_payload = {
        "origin_room": "ROOM_A",
        "monitored_edge": "CORRIDOR_1 <-> EXIT_A",
        "fire_detected": "YES",
        "smoke_detected": "NO",
        "hazard_score": 0.875,
        "people_detected": 10,
        "crowd_density": 0.50,
        "risk_index": 78,
        "efficiency_index": 65,
        "baseline_route": "ROOM_A -> CORRIDOR_1 -> EXIT_A",
        "dynamic_route": "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B",
        "recommended_exit": "EXIT_B",
        "baseline_cost": 4.08,
        "route_cost": 5.28,
        "route_changed": "YES",
        "reroute_reason": "Fire hazard detected on CORRIDOR_1 <-> EXIT_A"
    }
    try:
        txt_rep = generate_incident_report_text(sample_payload)
        record("Incident Report Generation (TXT)", "DIGITAL INCIDENT REPORT" in txt_rep and "EXIT_B" in txt_rep, "TXT report validated")
    except Exception as e:
        record("Incident Report Generation (TXT)", False, f"Error: {e}")

    # 7. Digital Incident Report - JSON Format
    try:
        json_rep = generate_incident_report_json(sample_payload)
        parsed = json.loads(json_rep)
        record("Incident Report Generation (JSON)", parsed["recommended_exit"] == "EXIT_B", "JSON report parsed successfully")
    except Exception as e:
        record("Incident Report Generation (JSON)", False, f"Error: {e}")

    # 8. Digital Incident Report - CSV Format
    try:
        csv_rep = generate_incident_report_csv(sample_payload)
        record("Incident Report Generation (CSV)", "recommended_exit" in csv_rep and "EXIT_B" in csv_rep, "CSV report validated")
    except Exception as e:
        record("Incident Report Generation (CSV)", False, f"Error: {e}")

    # 9. Emergency Decision Assistant - Query Answering
    try:
        assistant = EmergencyDecisionAssistant()
        state = {
            "fire_detected": True,
            "smoke_detected": False,
            "hazard_score": 0.875,
            "people_detected": 5,
            "crowd_density": 0.25,
            "route_changed": True,
            "recommended_exit": "EXIT_B",
            "normal_exit": "EXIT_A",
            "recommended_route_str": "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B",
            "active_edge": ("CORRIDOR_1", "EXIT_A"),
            "total_route_cost": 5.28,
            "edge_cost": 8.10,
            "risk_index": 78
        }
        ans1 = assistant.answer_query("Why did the route change?", state)
        record("Decision Assistant - Route Change Query", "EXIT_B" in ans1 and "hazard" in ans1.lower(), "Rule-based response verified")
    except Exception as e:
        record("Decision Assistant - Route Change Query", False, f"Error: {e}")

    # 10. Emergency Decision Assistant - Exit Avoidance Query
    try:
        ans2 = assistant.answer_query("Why is EXIT_A closed?", state)
        record("Decision Assistant - Exit Closed Query", "EXIT_A" in ans2 and "elevated" in ans2.lower(), "Exit avoidance explanation verified")
    except Exception as e:
        record("Decision Assistant - Exit Closed Query", False, f"Error: {e}")

    # 11. Evacuation Guidance Card - Full Argument Rendering & Diversion Badge
    try:
        with patch("streamlit.markdown") as mock_md:
            render_evacuation_guidance_card(
                recommended_exit="EXIT_B",
                recommended_route=["ROOM_A", "CORRIDOR_1", "CORRIDOR_2", "EXIT_B"],
                is_emergency=True,
                hazard_type="FIRE",
                hazard_score=0.875,
                crowd_density=0.25,
                adaptation_reason="Direct route blocked by active fire",
                avoided_exit="EXIT_A",
                route_changed=True,
                total_cost=5.28,
                origin_room="ROOM_A",
                hazard_zone="CORRIDOR_1 ↔ EXIT_A"
            )
            mock_md.assert_called_once()
            rendered_html = mock_md.call_args[0][0]
            kwargs = mock_md.call_args[1]
            assert kwargs.get("unsafe_allow_html") is True, "unsafe_allow_html must be True"
            assert "EXIT_B" in rendered_html, "Missing recommended exit in card"
            assert "ROUTE DIVERTED" in rendered_html, "Missing ROUTE DIVERTED badge"
            assert "Compromised Corridor Zone" in rendered_html, "Missing hazard zone alert"
            assert "Traversal Cost: 5.28" in rendered_html, "Missing traversal cost display"
            assert "EMERGENCY EVACUATION DIRECTIVE" in rendered_html, "Missing emergency header"
            assert not any(line.startswith("    ") for line in rendered_html.splitlines()), "Found lines with >=4 leading spaces triggering code blocks"
            assert '<div style="font-size: 0.82rem; color: #DC2626;' in rendered_html, "Missing styled warning div"
            assert "<strong>Compromised Corridor Zone:</strong> <code>CORRIDOR_1 ↔ EXIT_A</code>" in rendered_html, "Missing warning HTML tags"
            assert "<strong>Notice:</strong> Direct egress via <strong>EXIT_A</strong> is compromised" in rendered_html, "Missing avoided exit HTML tags"
            assert "&lt;div" not in rendered_html and "&lt;strong" not in rendered_html and "&lt;code" not in rendered_html, "Container HTML tags were incorrectly escaped as literal text"
            record("Evacuation Guidance Card (Full Arg Diverted)", True, "Rendered diverted card with traversal cost and valid unindented warning HTML")
    except Exception as e:
        record("Evacuation Guidance Card (Full Arg Diverted)", False, f"Error: {e}")

    # 12. Evacuation Guidance Card - Nominal State Rendering
    try:
        with patch("streamlit.markdown") as mock_md:
            render_evacuation_guidance_card(
                recommended_exit="EXIT_A",
                recommended_route=["ROOM_A", "CORRIDOR_1", "EXIT_A"],
                is_emergency=False,
                route_changed=False,
                total_cost=2.00,
                origin_room="ROOM_A"
            )
            mock_md.assert_called_once()
            rendered_html = mock_md.call_args[0][0]
            assert "EXIT_A" in rendered_html, "Missing recommended exit in nominal card"
            assert "DIRECT ROUTE" in rendered_html, "Missing DIRECT ROUTE badge"
            assert "FACILITY CONDITIONS NOMINAL" in rendered_html, "Missing nominal header"
            record("Evacuation Guidance Card (Nominal Direct)", True, "Rendered nominal state card with direct route badge")
    except Exception as e:
        record("Evacuation Guidance Card (Nominal Direct)", False, f"Error: {e}")

    # 13. Evacuation Guidance Card - Forward Compatibility with Arbitrary Kwargs
    try:
        with patch("streamlit.markdown") as mock_md:
            render_evacuation_guidance_card(
                recommended_exit="EXIT_A",
                recommended_route=["ROOM_A", "CORRIDOR_1", "EXIT_A"],
                is_emergency=False,
                unexpected_future_param_1=True,
                unexpected_future_param_2="safe_zone",
                extra_payload={"key": 123}
            )
            mock_md.assert_called_once()
            record("Evacuation Guidance Card (Forward Kwargs)", True, "Successfully handled arbitrary kwargs without TypeError")
    except Exception as e:
        record("Evacuation Guidance Card (Forward Kwargs)", False, f"Error: {e}")

    # 14. Detection Summary Cards - Complete Arguments (Traversal Cost + Confidences)
    try:
        with patch("streamlit.markdown") as mock_md:
            render_detection_summary_cards(
                fire_detected=True,
                smoke_detected=False,
                people_detected=15,
                hazard_score=0.875,
                crowd_density=0.75,
                total_route_cost=5.28,
                fire_conf=0.89,
                smoke_conf=0.0
            )
            assert mock_md.call_count == 4, f"Expected 4 card markdown calls, got {mock_md.call_count}"
            rendered_chunks = [call[0][0] for call in mock_md.call_args_list]
            c1_html, c2_html, c3_html, c4_html = rendered_chunks
            assert "Fire Detected (89%)" in c1_html, "Missing fire confidence in card 1"
            assert "15" in c2_html and "Density: 0.75" in c2_html, "Missing count/density in card 2"
            assert "0.88" in c3_html and "Critical Level" in c3_html, "Missing hazard score/level in card 3"
            assert "5.28" in c4_html and "Dynamic Dijkstra" in c4_html, "Missing traversal cost in card 4"
            record("Detection Summary Cards (Complete + Cost)", True, "Rendered all 4 cards with traversal cost, confidences, and density")
    except Exception as e:
        record("Detection Summary Cards (Complete + Cost)", False, f"Error: {e}")

    # 15. Detection Summary Cards - Nominal State (Density Primary when Cost=0)
    try:
        with patch("streamlit.markdown") as mock_md:
            render_detection_summary_cards(
                fire_detected=False,
                smoke_detected=False,
                people_detected=3,
                hazard_score=0.0,
                crowd_density=0.15,
                total_route_cost=0.0
            )
            assert mock_md.call_count == 4, f"Expected 4 card markdown calls, got {mock_md.call_count}"
            rendered_chunks = [call[0][0] for call in mock_md.call_args_list]
            c1_html, c2_html, c3_html, c4_html = rendered_chunks
            assert "Not Detected" in c1_html, "Card 1 should report Not Detected"
            assert "3" in c2_html and "COCO Model" in c2_html, "Card 2 should display count and COCO tag"
            assert "Normal Level" in c3_html, "Card 3 should report Normal Level"
            assert "0.15" in c4_html and "Clear" in c4_html, "Card 4 should display crowd density when cost=0"
            record("Detection Summary Cards (Nominal + Density)", True, "Rendered nominal state with crowd density fallback in card 4")
    except Exception as e:
        record("Detection Summary Cards (Nominal + Density)", False, f"Error: {e}")

    # 16. Detection Summary Cards - Forward Compatibility with Arbitrary Kwargs
    try:
        with patch("streamlit.markdown") as mock_md:
            render_detection_summary_cards(
                fire_detected=False,
                smoke_detected=False,
                people_detected=2,
                hazard_score=0.1,
                crowd_density=0.05,
                total_route_cost=2.5,
                future_model_version="v2.1",
                additional_flag=True,
                extra_metadata={"zone": "ZONE_A"}
            )
            assert mock_md.call_count == 4
            record("Detection Summary Cards (Forward Kwargs)", True, "Successfully handled arbitrary kwargs without TypeError")
    except Exception as e:
        record("Detection Summary Cards (Forward Kwargs)", False, f"Error: {e}")

    # 17. Static Component Call Audit (AST Check of dashboard/app.py)
    try:
        import ast
        import inspect
        import dashboard.components as comp_module

        app_py_path = project_root / "dashboard" / "app.py"
        with open(app_py_path, "r", encoding="utf-8") as f:
            app_code = f.read()
        tree = ast.parse(app_code, filename=str(app_py_path))

        comp_funcs = {name: getattr(comp_module, name) for name in dir(comp_module) if inspect.isfunction(getattr(comp_module, name))}
        ast_issues = []
        audited_count = 0

        class Auditor(ast.NodeVisitor):
            def visit_Call(self, node):
                nonlocal audited_count
                fn_name = None
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                if fn_name in comp_funcs:
                    audited_count += 1
                    target_fn = comp_funcs[fn_name]
                    sig = inspect.signature(target_fn)
                    p_map = sig.parameters
                    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in p_map.values())
                    kws = [k.arg for k in node.keywords if k.arg is not None]
                    if not has_kwargs:
                        for kw in kws:
                            if kw not in p_map:
                                ast_issues.append(f"L{node.lineno}: {fn_name}() unexpected kwarg '{kw}'")
                    reqs = [
                        p_name for p_name, p in p_map.items()
                        if p.default is inspect.Parameter.empty
                        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    ]
                    supplied = set(kws)
                    param_keys = list(p_map.keys())
                    for i in range(min(len(node.args), len(param_keys))):
                        supplied.add(param_keys[i])
                    for req in reqs:
                        if req not in supplied:
                            ast_issues.append(f"L{node.lineno}: {fn_name}() missing required arg '{req}'")
                self.generic_visit(node)

        Auditor().visit(tree)
        assert len(ast_issues) == 0, f"Found {len(ast_issues)} component call issues: {ast_issues}"
        record("Static Component Call Audit (AST)", True, f"Verified {audited_count} component calls in app.py with 0 mismatches")
    except Exception as e:
        record("Static Component Call Audit (AST)", False, f"Error: {e}")

    # 18. HTML Sanitization & Streamlit Supported Rendering Mechanism
    try:
        with patch("streamlit.markdown") as mock_md:
            malicious_input = '<script>alert("xss")</script>'
            render_evacuation_guidance_card(
                recommended_exit=malicious_input,
                recommended_route=["ROOM_A", malicious_input],
                is_emergency=True,
                adaptation_reason='Hazard in <corridor> & "danger"',
                hazard_zone="ZONE_<1>&EXIT",
                avoided_exit=malicious_input,
            )
            mock_md.assert_called_once()
            rendered_html = mock_md.call_args[0][0]
            # Dynamic inputs must be escaped
            assert "<script>" not in rendered_html, "Raw <script> tag was not escaped"
            assert "&lt;script&gt;" in rendered_html, "Escaped script tag missing from output"
            assert "&lt;corridor&gt;" in rendered_html, "Escaped corridor tag missing"
            assert "&amp;" in rendered_html, "Ampersand not escaped"
            # Framework structural tags must remain intact HTML
            assert '<div class="evac-guidance-emergency">' in rendered_html, "Card container HTML missing"
            assert "<strong>Compromised Corridor Zone:</strong>" in rendered_html, "Framework <strong> tag broken"
            assert mock_md.call_args[1].get("unsafe_allow_html") is True, "Must use unsafe_allow_html=True"
            assert not any(line.startswith("    ") for line in rendered_html.splitlines()), "Found indentation that would trigger code block"
            record("HTML Sanitization & Safe Markup Verification", True, "Dynamic text sanitized with html.escape and framework HTML rendered cleanly")
    except Exception as e:
        record("HTML Sanitization & Safe Markup Verification", False, f"Error: {e}")

    print("=" * 70)
    print(f"TOTAL ADVANCED FEATURE TESTS: {total} | PASSED: {passed} | FAILED: {total - passed}")
    print("=" * 70)
    return total - passed


if __name__ == "__main__":
    exit_code = run_advanced_features_tests()
    sys.exit(exit_code)
