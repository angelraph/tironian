

def test_gate_reasons_name_each_step_kind_and_never_round_up_to_the_floor():
    from plugin.core.evaluate import Check, gated_scorecard
    good = Check(passed=True, valid=True, reason="")
    bad = Check(passed=False, valid=True, reason="action_mismatch")
    broken = Check(passed=False, valid=False, reason="invalid")
    checks = [good] * 19 + [good] * 8 + [bad] * 21 + [broken]
    kinds = ["first"] * 19 + ["follow"] * 30
    sc = gated_scorecard(checks, [1] * len(checks), kinds)
    assert not sc.gate_passed
    assert "first steps: needs" in sc.gate_reason and "follow steps:" in sc.gate_reason
    assert "98% < 98%" not in sc.gate_reason
