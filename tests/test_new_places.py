import random

from plugin.core.evaluate import Check, apply_new_places, gated_scorecard
from plugin.core.families import place
from plugin.core.pipeline import _hold_out_places


def test_place_finds_the_folder_tree_or_drive():
    assert place("Find the 10 largest files under my Music directory") == "music"
    assert place(r"biggest files in my AppData\Local\Temp folder") == "appdata"
    assert place(r"How many pdfs in D:\Data\Reports?") == "d:/data"
    assert place(r"count zips in C:\Windows\Fonts") == "c:/windows"
    assert place(r"count pdfs in C:\Users\bob\Desktop\a") == "desktop"
    assert place("find /c/Users/bob/Videos -iname x") == "videos"
    assert place("ls ~/Downloads") == "downloads"
    assert place("what time is it") == ""


def test_whole_folder_trees_are_held_out():
    places = {f"s{i}": ["music", "videos", "downloads", "desktop", "pictures"][i % 5] for i in range(50)}
    sessions = sorted(places)
    held = _hold_out_places(sessions, places, n_hold=10, cap=20, rng=random.Random(1))
    assert held and len(held) >= 10
    held_trees = {places[s] for s in held}
    train_trees = {places[s] for s in sessions if s not in held}
    assert not held_trees & train_trees     # no tree is on both sides


def test_the_test_keeps_both_home_and_other_trees():
    trees = ["music", "videos", "downloads", "desktop", "c:/windows", "c:/program files", "c:/tools", "d:/data"]
    places = {f"s{i:02}": trees[i % len(trees)] for i in range(80)}
    sessions = sorted(places)
    held = _hold_out_places(sessions, places, n_hold=16, cap=32, rng=random.Random(3))
    held_trees = {places[s] for s in held}
    train_trees = {places[s] for s in sessions if s not in held}
    is_home = {"music", "videos", "downloads", "desktop"}
    assert held_trees & is_home and held_trees - is_home       # the test has both kinds
    assert train_trees & is_home and train_trees - is_home     # so does training


def test_too_few_trees_falls_back_to_the_session_split():
    places = {f"s{i}": "music" if i % 2 else "videos" for i in range(20)}
    assert _hold_out_places(sorted(places), places, 4, 8, random.Random(1)) is None


def test_new_folders_can_stop_a_kind_of_step_but_not_add_one():
    good, bad = Check(True, True), Check(False, True, reason="action_mismatch")
    # 110 first steps: 100 on trained folders all right, 10 on a new drive with 5 wrong.
    checks = [good] * 100 + [good] * 5 + [bad] * 5
    kinds = ["first"] * 110
    places = ["music"] * 100 + ["d:/data"] * 10
    sc = gated_scorecard(checks, [1] * 110, kinds)
    assert sc.gate_passed and sc.serve_steps == ["first"]   # overall, it passes
    apply_new_places(sc, checks, kinds, places, {"music"}, threshold=0.85, min_n=10)
    assert sc.new_places["steps"]["first"] == {"n": 10, "passed": 5, "wilson_lo": sc.new_places["steps"]["first"]["wilson_lo"]}
    assert sc.serve_steps == [] and not sc.gate_passed
    assert "never trained on" in sc.gate_reason


def test_new_folders_that_hold_up_change_nothing():
    good = Check(True, True)
    checks, kinds = [good] * 40, ["first"] * 40
    places = ["music"] * 30 + ["d:/data"] * 10
    sc = gated_scorecard(checks, [1] * 40, kinds)
    apply_new_places(sc, checks, kinds, places, {"music"}, threshold=0.85, min_n=10)
    assert sc.gate_passed and sc.serve_steps == ["first"]
    assert sc.new_places["trees"] == ["d:/data"]
