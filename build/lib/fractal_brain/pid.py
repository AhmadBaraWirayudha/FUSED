"""
fractal_brain/pid.py
Discrete PID controller.
No external dependencies.
"""


class PIDController:
    """A simple discrete PID controller with integral anti‑windup via clamping."""
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0.0,
                 integral_min: float = -10.0, integral_max: float = 10.0, history_window: int = 50):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.prev_error = 0.0
        self.integral = 0.0
        # breakdown of the most recent step(), cached for compute_output() below
        self.last_error = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0
        # rolling history for stability_report(): a plain list manually capped at
        # history_window (not collections.deque -- checkpoint.py's serializer walks
        # plain dict/list/float/str/None and every registered class, and doesn't know
        # about deque; this keeps stability_report() working with the existing
        # checkpoint/restore path with no changes needed there).
        self.history_window = history_window
        self._history = []

    def step(self, error: float, dt: float = 1.0) -> float:
        """
        Compute PID output given the current error signal.

        `error` should be the raw deviation you want driven to zero (e.g.
        measurement - target). It is compared against `self.setpoint` internally
        (e = error - setpoint); if you've already computed a deviation yourself,
        just leave setpoint at its default of 0 and pass that deviation straight in.
        """
        e = error - self.setpoint
        self.integral += e * dt
        # Clamp integral to avoid windup
        saturated = False
        if self.integral > self.integral_max:
            self.integral = self.integral_max
            saturated = True
        elif self.integral < self.integral_min:
            self.integral = self.integral_min
            saturated = True

        derivative = (e - self.prev_error) / dt if dt else 0.0
        output = self.Kp * e + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = e
        self.last_error, self.last_integral, self.last_derivative = e, self.integral, derivative
        self._history.append((e, output, saturated))
        if len(self._history) > self.history_window:
            self._history.pop(0)
        return output

    def compute_output(self, Kp=None, Ki=None, Kd=None) -> float:
        """
        Recompute the PID formula from the *cached* (error, integral, derivative) of the
        most recent step(), optionally substituting different gains -- without touching
        any internal state (integral, prev_error). Useful for probing "what would the
        output have been with a slightly different gain", e.g. for finite-difference
        gradient estimation, without corrupting the controller's real trajectory.
        """
        Kp = self.Kp if Kp is None else Kp
        Ki = self.Ki if Ki is None else Ki
        Kd = self.Kd if Kd is None else Kd
        return Kp * self.last_error + Ki * self.last_integral + Kd * self.last_derivative

    def stability_report(self, oscillation_ratio_threshold: float = 1.5) -> dict:
        """
        A practical, empirical stability diagnostic over the recent rolling history
        (up to history_window steps), for a controller whose downstream use (e.g.
        modulating a gate temperature through a nonlinear squash, itself feeding back
        into the next step's error) is a nonlinear, time-varying loop -- there's no
        fixed linear "plant matrix" here to eigendecompose, so this checks for the
        actual, observable symptoms of instability instead of computing eigenvalues
        of something that doesn't exist in this architecture:

        - integral_saturation_fraction: fraction of recent steps where the integral
          term was pinned at integral_min/integral_max. Persistently near 1.0 means
          the gains are pushing the integral term harder than the anti-windup clamp
          allows, every step -- a classic sign of poorly-scaled gains, not a
          momentary, healthy response to a real setpoint change.
        - oscillation_ratio: mean |output| over the second half of the window
          divided by the mean |output| over the first half. Meaningfully > 1 means
          amplitude is *growing* over the window, not just varying -- the practical
          signature of instability in a feedback loop, versus a value near 1 (stable
          or decaying oscillation) or well under 1 (settling down).
        - is_stable: oscillation_ratio below oscillation_ratio_threshold AND
          integral_saturation_fraction below 0.8. Both are heuristic thresholds, not
          derived bounds -- treat this as "worth a closer look", not a proof.
        - insufficient_history: True (and the other fields None) if step() hasn't
          been called enough times yet to say anything meaningful.
        """
        n = len(self._history)
        if n < 10:
            return {"insufficient_history": True, "steps_recorded": n,
                     "integral_saturation_fraction": None, "oscillation_ratio": None, "is_stable": None}

        saturated_count = sum(1 for _, _, sat in self._history if sat)
        integral_saturation_fraction = saturated_count / n

        half = n // 2
        first_half_outputs = [abs(o) for _, o, _ in self._history[:half]]
        second_half_outputs = [abs(o) for _, o, _ in self._history[half:]]
        mean_first = sum(first_half_outputs) / len(first_half_outputs) if first_half_outputs else 0.0
        mean_second = sum(second_half_outputs) / len(second_half_outputs) if second_half_outputs else 0.0
        oscillation_ratio = (mean_second / mean_first) if mean_first > 1e-9 else (1.0 if mean_second < 1e-9 else float("inf"))

        is_stable = (oscillation_ratio < oscillation_ratio_threshold) and (integral_saturation_fraction < 0.8)

        return {
            "insufficient_history": False,
            "steps_recorded": n,
            "integral_saturation_fraction": integral_saturation_fraction,
            "oscillation_ratio": oscillation_ratio,
            "is_stable": is_stable,
        }

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0
        self._history = []