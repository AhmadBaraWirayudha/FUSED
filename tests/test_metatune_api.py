from metatune import Categorical, Float, Int, Study


def _objective(cfg):
    return -((cfg["x"] - 0.3) ** 2 + (cfg["y"] - 0.7) ** 2)


def test_optimize_convenience_finds_the_optimum_reasonably_well():
    study = Study(space={"x": Float(-2, 2), "y": Float(-2, 2)}, direction="maximize")
    study.optimize(_objective, n_trials=40)
    assert study.best_value is not None
    assert study.best_value > -0.5  # comfortably better than a random guess


def test_manual_ask_tell_matches_optimize_shape():
    study = Study(
        space={"x": Float(-2, 2), "y": Float(-2, 2), "label": Categorical(["a", "b"]), "n": Int(1, 5)},
        direction="maximize",
    )
    for _ in range(30):
        config = study.ask()
        assert set(config) == {"x", "y", "label", "n"}
        study.tell(_objective(config))
    assert study.best_config is not None
    assert study.best_config["label"] in ("a", "b")
    assert 1 <= study.best_config["n"] <= 5


def test_minimize_direction():
    study = Study(space={"x": Float(-2, 2)}, direction="minimize")
    study.optimize(lambda cfg: cfg["x"] ** 2, n_trials=30)
    assert abs(study.best_config["x"]) < 1.0


def test_tell_without_ask_raises():
    study = Study(space={"x": Float(-2, 2)}, direction="maximize")
    try:
        study.tell(1.0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
