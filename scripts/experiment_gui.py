"""ML Autoscaler — Experiment Dashboard (CustomTkinter).

Interactive visualization of:
  - Neural network internals (buffer, features, ONNX inference)
  - Scaling algorithm 9-step pipeline
  - HPA vs ML autoscaler comparison

Usage:
    python scripts/experiment_gui.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import customtkinter as ctk
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.experiment_scenarios import get_all_scenarios, Scenario
from scripts.experiment_simulator import (
    ExperimentRunner,
    HPAConfig,
    MLAutoscalerSimulator,
    MLConfig,
    StepRecord,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Color palette (unified with Grafana dashboards)
# ---------------------------------------------------------------------------

C = {
    "bg":        "#1a1a2e",
    "bg_card":   "#16213e",
    "sidebar":   "#0f3460",
    "accent":    "#e94560",
    "green":     "#73BF69",
    "yellow":    "#FADE2A",
    "orange":    "#FF9830",
    "red":       "#F2495C",
    "blue":      "#5794F2",
    "purple":    "#B877D9",
    "text":      "#e0e0e0",
    "text_dim":  "#8892b0",
    "grid":      "#2a2a4a",
}

# Matplotlib dark style
plt.rcParams.update({
    "figure.facecolor": C["bg_card"],
    "axes.facecolor": C["bg"],
    "axes.edgecolor": C["grid"],
    "axes.labelcolor": C["text"],
    "text.color": C["text"],
    "xtick.color": C["text_dim"],
    "ytick.color": C["text_dim"],
    "grid.color": C["grid"],
    "grid.alpha": 0.3,
    "font.size": 9,
    "legend.facecolor": C["bg_card"],
    "legend.edgecolor": C["grid"],
})


# ---------------------------------------------------------------------------
# Utility: embed a matplotlib figure in a CTkFrame
# ---------------------------------------------------------------------------


def embed_figure(parent: ctk.CTkFrame, fig: Figure) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# ---------------------------------------------------------------------------
# KPI Card widget
# ---------------------------------------------------------------------------


class KPICard(ctk.CTkFrame):
    """Compact KPI card: title + HPA value + ML value."""

    def __init__(self, master, title: str, **kw):
        super().__init__(master, fg_color=C["bg_card"], corner_radius=8, **kw)
        self._title_lbl = ctk.CTkLabel(
            self, text=title, font=("Consolas", 11), text_color=C["text_dim"]
        )
        self._title_lbl.pack(pady=(6, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=(0, 6))
        self._hpa_lbl = ctk.CTkLabel(
            row, text="HPA: --", font=("Consolas", 13, "bold"), text_color=C["red"]
        )
        self._hpa_lbl.pack(side="left", padx=6)
        self._ml_lbl = ctk.CTkLabel(
            row, text="ML: --", font=("Consolas", 13, "bold"), text_color=C["blue"]
        )
        self._ml_lbl.pack(side="left", padx=6)

    def set_values(self, hpa_val: str, ml_val: str):
        self._hpa_lbl.configure(text=f"HPA: {hpa_val}")
        self._ml_lbl.configure(text=f"ML: {ml_val}")


# ---------------------------------------------------------------------------
# Pipeline Step block
# ---------------------------------------------------------------------------


class PipelineBlock(ctk.CTkFrame):
    """One step in the 9-step scaling algorithm visualization."""

    def __init__(self, master, step_num: int, title: str, **kw):
        super().__init__(master, fg_color=C["bg_card"], corner_radius=6,
                         border_width=1, border_color=C["grid"], **kw)
        self.configure(width=115, height=80)
        self.pack_propagate(False)
        ctk.CTkLabel(
            self, text=f"Step {step_num}", font=("Consolas", 8),
            text_color=C["text_dim"]
        ).pack(pady=(4, 0))
        ctk.CTkLabel(
            self, text=title, font=("Consolas", 9, "bold"),
            text_color=C["text"]
        ).pack()
        self._value_lbl = ctk.CTkLabel(
            self, text="—", font=("Consolas", 10), text_color=C["text_dim"],
            wraplength=100
        )
        self._value_lbl.pack(pady=(2, 4))

    def update(self, text: str, status: str = "active"):
        colors = {
            "active": C["green"],
            "blocked": C["yellow"],
            "skipped": C["text_dim"],
            "executed": C["accent"],
        }
        self._value_lbl.configure(text=text, text_color=colors.get(status, C["text"]))
        border = {
            "active": C["green"],
            "blocked": C["yellow"],
            "executed": C["accent"],
        }
        self.configure(border_color=border.get(status, C["grid"]))


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class ExperimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ML Autoscaler — Experiment Dashboard")
        self.geometry("1440x920")
        ctk.set_appearance_mode("dark")

        # State
        self._scenarios: list[Scenario] = []
        self._current_scenario_idx = 0
        self._step = 0
        self._playing = False
        self._speed_ms = 200
        self._history: list[StepRecord] = []

        # ML simulator state for step-by-step
        self._ml_sim: MLAutoscalerSimulator | None = None
        self._hpa_replicas_trace: list[int] = []
        self._ml_replicas_trace: list[int] = []
        self._actual_trace: list[float] = []
        self._predicted_trace: list[float] = []
        self._smoothed_trace: list[float] = []
        self._decisions_log: list[str] = []

        # Build UI
        self._build_controls()
        self._build_sidebar()
        self._build_tabview()

        # Load data
        self._load_scenarios()

    # -------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------

    def _load_scenarios(self):
        try:
            self._scenarios = get_all_scenarios(
                ROOT / "data" / "processed" / "test.csv",
                ROOT / "data" / "processed" / "scaler.pkl",
                ROOT / "data" / "processed" / "metadata.json",
            )
        except Exception as e:
            self._scenarios = [self._synthetic_fallback()]
        self._select_scenario(0)

    def _synthetic_fallback(self) -> Scenario:
        """Create a simple synthetic scenario if data files are missing."""
        n = 200
        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, n)
        actual = 50 + 25 * np.sin(t) + rng.normal(0, 3, n)
        predicted = np.roll(actual, -3)
        predicted[-3:] = actual[-3:]
        predicted += rng.normal(0, 2, n)
        return Scenario(
            name="synthetic_demo",
            description="Synthetic sine wave demo",
            actual_cpu=np.clip(actual, 10, 100),
            predicted_cpu=np.clip(predicted, 10, 100),
            duration_hours=n * 5 / 60,
        )

    def _select_scenario(self, idx: int):
        self._current_scenario_idx = idx
        self._reset_simulation()
        self._update_all()

    def _reset_simulation(self):
        self._step = 0
        self._playing = False
        sc = self._scenarios[self._current_scenario_idx]

        # Run full experiment for Tab 3
        runner = ExperimentRunner(
            sc.actual_cpu, sc.predicted_cpu,
            ml_config=self._build_ml_config(),
        )
        self._history, self._full_metrics = runner.run()

        # Reset step-by-step state
        self._ml_sim = MLAutoscalerSimulator(self._build_ml_config())
        self._hpa_replicas_trace = []
        self._ml_replicas_trace = []
        self._actual_trace = []
        self._predicted_trace = []
        self._smoothed_trace = []
        self._decisions_log = []

    def _build_ml_config(self) -> MLConfig:
        return MLConfig(
            target_utilization=self._target_var.get(),
            smoothing_alpha=self._alpha_var.get(),
            safety_margin=self._safety_var.get(),
            scale_up_threshold_factor=self._up_thresh_var.get(),
            scale_down_threshold_factor=self._down_thresh_var.get(),
        )

    # -------------------------------------------------------------------
    # Control bar (top)
    # -------------------------------------------------------------------

    def _build_controls(self):
        bar = ctk.CTkFrame(self, fg_color=C["bg_card"], height=50)
        bar.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            bar, text="ML Autoscaler — Experiment Dashboard",
            font=("Consolas", 16, "bold"), text_color=C["text"]
        ).pack(side="left", padx=12)

        # Transport controls
        self._play_btn = ctk.CTkButton(
            bar, text="▶  Play", width=90, fg_color=C["green"],
            text_color="#000", hover_color="#5ea554",
            command=self._toggle_play
        )
        self._play_btn.pack(side="left", padx=4)

        ctk.CTkButton(
            bar, text="⏭  Step", width=90, fg_color=C["blue"],
            text_color="#000", hover_color="#4678c9",
            command=self._do_step
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            bar, text="↺  Reset", width=90, fg_color=C["orange"],
            text_color="#000", hover_color="#d97e24",
            command=lambda: (self._reset_simulation(), self._update_all())
        ).pack(side="left", padx=4)

        # Step counter
        self._step_lbl = ctk.CTkLabel(
            bar, text="Step: 0 / 0", font=("Consolas", 12),
            text_color=C["text_dim"]
        )
        self._step_lbl.pack(side="right", padx=12)

        # Speed slider
        ctk.CTkLabel(bar, text="Speed:", text_color=C["text_dim"]).pack(side="right")
        self._speed_slider = ctk.CTkSlider(
            bar, from_=50, to=1000, width=120,
            command=lambda v: setattr(self, "_speed_ms", int(v))
        )
        self._speed_slider.set(200)
        self._speed_slider.pack(side="right", padx=4)

    def _toggle_play(self):
        self._playing = not self._playing
        self._play_btn.configure(
            text="⏸  Pause" if self._playing else "▶  Play",
            fg_color=C["yellow"] if self._playing else C["green"],
        )
        if self._playing:
            self._auto_step()

    def _auto_step(self):
        if not self._playing:
            return
        sc = self._scenarios[self._current_scenario_idx]
        if self._step < len(sc.actual_cpu):
            self._do_step()
            self.after(self._speed_ms, self._auto_step)
        else:
            self._playing = False
            self._play_btn.configure(text="▶  Play", fg_color=C["green"])

    def _do_step(self):
        sc = self._scenarios[self._current_scenario_idx]
        if self._step >= len(sc.actual_cpu):
            return

        t = self._step
        actual = float(sc.actual_cpu[t])
        predicted = float(sc.predicted_cpu[t])

        # ML autoscaler step
        ml_replicas, ml_dir = self._ml_sim.step(predicted, t)
        smoothed = self._ml_sim._smoothed or predicted

        # HPA (from pre-computed history)
        hpa_replicas = self._history[t].hpa_replicas if t < len(self._history) else 2

        self._actual_trace.append(actual)
        self._predicted_trace.append(predicted)
        self._smoothed_trace.append(smoothed)
        self._ml_replicas_trace.append(ml_replicas)
        self._hpa_replicas_trace.append(hpa_replicas)
        self._decisions_log.append(
            f"[t={t:>3}] {ml_dir:>4} | ML:{ml_replicas} HPA:{hpa_replicas} | "
            f"actual={actual:.1f}% pred={predicted:.1f}% smooth={smoothed:.1f}%"
        )

        self._step += 1
        self._update_all()

    # -------------------------------------------------------------------
    # Sidebar (parameters)
    # -------------------------------------------------------------------

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=C["sidebar"], width=220, corner_radius=0)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar, text="Parameters", font=("Consolas", 14, "bold"),
            text_color=C["text"]
        ).pack(pady=(12, 8))

        def make_slider(parent, label, from_, to, default, resolution=None):
            ctk.CTkLabel(
                parent, text=label, font=("Consolas", 10), text_color=C["text_dim"]
            ).pack(pady=(8, 0), padx=8, anchor="w")
            var = ctk.DoubleVar(value=default)
            val_lbl = ctk.CTkLabel(
                parent, text=f"{default}", font=("Consolas", 10), text_color=C["text"]
            )
            val_lbl.pack(padx=8, anchor="e")

            def on_change(v):
                val_lbl.configure(text=f"{float(v):.2f}")

            slider = ctk.CTkSlider(
                parent, from_=from_, to=to, variable=var,
                width=180, command=on_change
            )
            slider.pack(padx=8)
            return var

        self._target_var = make_slider(sidebar, "Target Util %", 20, 90, 60)
        self._alpha_var = make_slider(sidebar, "EMA Alpha", 0.1, 1.0, 0.7)
        self._safety_var = make_slider(sidebar, "Safety Margin", 0.0, 0.5, 0.15)
        self._up_thresh_var = make_slider(sidebar, "Up Threshold ×", 1.0, 1.5, 1.1)
        self._down_thresh_var = make_slider(sidebar, "Down Threshold ×", 0.3, 0.9, 0.7)

        ctk.CTkButton(
            sidebar, text="Apply & Re-run", fg_color=C["accent"],
            text_color="#fff", hover_color="#c73550",
            command=lambda: (self._reset_simulation(), self._update_all())
        ).pack(pady=16, padx=12, fill="x")

        # Scenario selector
        ctk.CTkLabel(
            sidebar, text="Scenario", font=("Consolas", 14, "bold"),
            text_color=C["text"]
        ).pack(pady=(16, 4))

        self._scenario_btns: list[ctk.CTkButton] = []
        names = ["Stable", "Ramp", "Spike", "Diurnal"]
        for i, name in enumerate(names):
            btn = ctk.CTkButton(
                sidebar, text=name, width=180, height=28,
                fg_color=C["bg_card"], hover_color=C["blue"],
                command=lambda idx=i: self._select_scenario(idx)
            )
            btn.pack(pady=2, padx=12)
            self._scenario_btns.append(btn)

    # -------------------------------------------------------------------
    # Tab view (3 tabs)
    # -------------------------------------------------------------------

    def _build_tabview(self):
        self._tabview = ctk.CTkTabview(self, fg_color=C["bg"])
        self._tabview.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._tab_nn = self._tabview.add("Neural Network")
        self._tab_algo = self._tabview.add("Scaling Algorithm")
        self._tab_exp = self._tabview.add("Experiment")

        self._build_nn_tab()
        self._build_algo_tab()
        self._build_exp_tab()

    # --- Tab 1: Neural Network ---

    def _build_nn_tab(self):
        tab = self._tab_nn

        # Top: pipeline diagram
        pipe_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8, height=90)
        pipe_frame.pack(fill="x", padx=8, pady=4)
        pipe_frame.pack_propagate(False)

        ctk.CTkLabel(
            pipe_frame, text="ONNX Inference Pipeline",
            font=("Consolas", 12, "bold"), text_color=C["text"]
        ).pack(pady=(6, 4))

        blocks_frame = ctk.CTkFrame(pipe_frame, fg_color="transparent")
        blocks_frame.pack()
        stages = [
            ("Raw Metrics", "5 values", C["green"]),
            ("→", "", C["text_dim"]),
            ("Feature Eng.", "44 features", C["purple"]),
            ("→", "", C["text_dim"]),
            ("StandardScaler", "normalize", C["orange"]),
            ("→", "", C["text_dim"]),
            ("ONNX LSTM", "(1,24,44)→(1,1)", C["blue"]),
            ("→", "", C["text_dim"]),
            ("Inverse Scale", "→ CPU %", C["green"]),
        ]
        for title, sub, color in stages:
            if title == "→":
                ctk.CTkLabel(
                    blocks_frame, text="→", font=("Consolas", 16, "bold"),
                    text_color=C["text_dim"]
                ).pack(side="left", padx=2)
            else:
                f = ctk.CTkFrame(blocks_frame, fg_color=C["bg"], corner_radius=6,
                                 border_width=1, border_color=color)
                f.pack(side="left", padx=4)
                ctk.CTkLabel(f, text=title, font=("Consolas", 9, "bold"),
                             text_color=color).pack(padx=6, pady=(2, 0))
                ctk.CTkLabel(f, text=sub, font=("Consolas", 8),
                             text_color=C["text_dim"]).pack(padx=6, pady=(0, 2))

        # Middle: prediction output + buffer
        mid = ctk.CTkFrame(tab, fg_color="transparent")
        mid.pack(fill="x", padx=8, pady=4)

        # Buffer status
        buf_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8, width=250)
        buf_frame.pack(side="left", padx=(0, 8), fill="y")
        buf_frame.pack_propagate(False)
        ctk.CTkLabel(buf_frame, text="Buffer Status", font=("Consolas", 11, "bold"),
                     text_color=C["text"]).pack(pady=(8, 4))
        self._buf_progress = ctk.CTkProgressBar(buf_frame, width=200)
        self._buf_progress.pack(pady=4, padx=12)
        self._buf_progress.set(0)
        self._buf_lbl = ctk.CTkLabel(
            buf_frame, text="0 / 24  NOT READY", font=("Consolas", 10),
            text_color=C["red"]
        )
        self._buf_lbl.pack()

        # Prediction big number
        pred_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8, width=200)
        pred_frame.pack(side="left", padx=8, fill="y")
        pred_frame.pack_propagate(False)
        ctk.CTkLabel(pred_frame, text="ML Prediction", font=("Consolas", 11, "bold"),
                     text_color=C["text"]).pack(pady=(8, 2))
        self._pred_big = ctk.CTkLabel(
            pred_frame, text="—", font=("Consolas", 28, "bold"),
            text_color=C["blue"]
        )
        self._pred_big.pack(pady=4)
        self._pred_actual = ctk.CTkLabel(
            pred_frame, text="Actual: —", font=("Consolas", 10),
            text_color=C["text_dim"]
        )
        self._pred_actual.pack()

        # Raw metrics panel
        raw_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8)
        raw_frame.pack(side="left", fill="both", expand=True, padx=8)
        ctk.CTkLabel(raw_frame, text="Current Metrics", font=("Consolas", 11, "bold"),
                     text_color=C["text"]).pack(pady=(8, 4))
        self._raw_labels: dict[str, ctk.CTkLabel] = {}
        for metric in ["cpu_usage", "mem_util", "net_in", "net_out", "disk_io"]:
            lbl = ctk.CTkLabel(raw_frame, text=f"{metric}: —",
                               font=("Consolas", 10), text_color=C["text_dim"])
            lbl.pack(anchor="w", padx=12)
            self._raw_labels[metric] = lbl

        # Bottom: prediction history chart
        self._nn_fig, self._nn_ax = plt.subplots(1, 1, figsize=(10, 3), dpi=80)
        self._nn_canvas = embed_figure(tab, self._nn_fig)

    # --- Tab 2: Scaling Algorithm ---

    def _build_algo_tab(self):
        tab = self._tab_algo

        # 9-step pipeline
        pipe = ctk.CTkFrame(tab, fg_color="transparent")
        pipe.pack(fill="x", padx=4, pady=4)

        step_names = [
            "EMA", "Hysteresis", "Desired", "Clamp",
            "Rate Limit", "Cooldown", "Stabilize", "No Change?", "Commit"
        ]
        self._algo_blocks: list[PipelineBlock] = []
        for i, name in enumerate(step_names):
            if i > 0:
                ctk.CTkLabel(pipe, text="→", font=("Consolas", 14, "bold"),
                             text_color=C["text_dim"]).pack(side="left", padx=1)
            block = PipelineBlock(pipe, i + 1, name)
            block.pack(side="left", padx=2)
            self._algo_blocks.append(block)

        # State panel
        state_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8, height=40)
        state_frame.pack(fill="x", padx=8, pady=4)
        self._state_lbl = ctk.CTkLabel(
            state_frame,
            text="Cooldown UP: — | Cooldown DOWN: — | Consec. Below: — | Direction: —",
            font=("Consolas", 10), text_color=C["text_dim"]
        )
        self._state_lbl.pack(pady=6)

        # Charts: CPU + Replicas side by side
        charts = ctk.CTkFrame(tab, fg_color="transparent")
        charts.pack(fill="both", expand=True, padx=4, pady=4)

        left = ctk.CTkFrame(charts, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True)
        self._algo_cpu_fig, self._algo_cpu_ax = plt.subplots(1, 1, figsize=(5.5, 3), dpi=80)
        self._algo_cpu_canvas = embed_figure(left, self._algo_cpu_fig)

        right = ctk.CTkFrame(charts, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)
        self._algo_rep_fig, self._algo_rep_ax = plt.subplots(1, 1, figsize=(5.5, 3), dpi=80)
        self._algo_rep_canvas = embed_figure(right, self._algo_rep_fig)

        # Decision log
        self._decision_log = ctk.CTkTextbox(
            tab, height=100, font=("Consolas", 9), fg_color=C["bg"],
            text_color=C["text_dim"]
        )
        self._decision_log.pack(fill="x", padx=8, pady=(0, 4))

    # --- Tab 3: Experiment Comparison ---

    def _build_exp_tab(self):
        tab = self._tab_exp

        # Main comparison chart
        self._exp_fig, self._exp_axes = plt.subplots(2, 1, figsize=(10, 5), dpi=80,
                                                      gridspec_kw={"height_ratios": [2, 1]})
        self._exp_canvas = embed_figure(tab, self._exp_fig)

        # KPI cards grid
        cards_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=8, pady=4)

        kpi_names = [
            "Avg Replicas", "Replica-Min", "SLA Violations", "Scaling Events",
            "Over-Prov %", "Under-Prov %", "Reaction Time", "Cost Savings %"
        ]
        self._kpi_cards: list[KPICard] = []
        for i, name in enumerate(kpi_names):
            card = KPICard(cards_frame, name)
            card.grid(row=i // 4, column=i % 4, padx=4, pady=4, sticky="nsew")
            self._kpi_cards.append(card)
        for c in range(4):
            cards_frame.columnconfigure(c, weight=1)

    # -------------------------------------------------------------------
    # Update all visuals
    # -------------------------------------------------------------------

    def _update_all(self):
        sc = self._scenarios[self._current_scenario_idx]
        total = len(sc.actual_cpu)
        self._step_lbl.configure(text=f"Step: {self._step} / {total}")

        # Highlight active scenario button
        for i, btn in enumerate(self._scenario_btns):
            btn.configure(
                fg_color=C["blue"] if i == self._current_scenario_idx else C["bg_card"]
            )

        self._update_nn_tab()
        self._update_algo_tab()
        self._update_exp_tab()

    def _update_nn_tab(self):
        n = self._step

        # Buffer status
        buf_fill = min(n, 24)
        self._buf_progress.set(buf_fill / 24)
        ready = n >= 24
        self._buf_lbl.configure(
            text=f"{min(n, 54)} / 24  {'READY' if ready else 'FILLING...'}",
            text_color=C["green"] if ready else C["red"]
        )

        # Current prediction
        if n > 0:
            pred = self._predicted_trace[-1]
            actual = self._actual_trace[-1]
            color = C["green"] if pred < 42 else C["yellow"] if pred < 66 else C["orange"] if pred < 85 else C["red"]
            self._pred_big.configure(text=f"{pred:.1f}%", text_color=color)
            self._pred_actual.configure(text=f"Actual: {actual:.1f}%")

            # Raw metrics (simulated from CPU)
            self._raw_labels["cpu_usage"].configure(text=f"cpu_usage: {actual:.1f}%")
            self._raw_labels["mem_util"].configure(text=f"mem_util: {actual * 0.6:.1f}%")
            self._raw_labels["net_in"].configure(text=f"net_in: {actual * 1.2:.0f} KB/s")
            self._raw_labels["net_out"].configure(text=f"net_out: {actual * 0.8:.0f} KB/s")
            self._raw_labels["disk_io"].configure(text=f"disk_io: {actual * 0.3:.1f}%")

        # Prediction history chart
        self._nn_ax.clear()
        if n > 1:
            x = range(max(0, n - 80), n)
            a_slice = self._actual_trace[max(0, n - 80):]
            p_slice = self._predicted_trace[max(0, n - 80):]
            self._nn_ax.plot(x, a_slice, color=C["green"], lw=1.5, label="Actual CPU")
            self._nn_ax.plot(x, p_slice, color=C["purple"], lw=1.5, ls="--", label="Predicted")
            self._nn_ax.axhline(60, color=C["yellow"], ls=":", lw=0.8, alpha=0.5)
            self._nn_ax.set_ylabel("CPU %")
            self._nn_ax.set_ylim(0, 100)
            self._nn_ax.legend(loc="upper right", fontsize=8)
            self._nn_ax.grid(True, alpha=0.2)
        self._nn_ax.set_title("Prediction History (last 80 steps)", fontsize=10)
        self._nn_fig.tight_layout()
        self._nn_canvas.draw_idle()

    def _update_algo_tab(self):
        n = self._step
        if n == 0:
            for block in self._algo_blocks:
                block.update("—", "skipped")
            return

        # Get current step details
        pred = self._predicted_trace[-1]
        smoothed = self._smoothed_trace[-1]
        ml_replicas = self._ml_replicas_trace[-1]
        prev_replicas = self._ml_replicas_trace[-2] if n > 1 else 2

        target = self._target_var.get()
        up_thresh = target * self._up_thresh_var.get()
        down_thresh = target * self._down_thresh_var.get()

        # Determine direction
        if smoothed > up_thresh:
            direction = "UP"
        elif smoothed < down_thresh:
            direction = "DOWN"
        else:
            direction = "HOLD"

        # Update blocks
        self._algo_blocks[0].update(f"{pred:.1f}→{smoothed:.1f}%", "active")
        self._algo_blocks[1].update(
            f"{smoothed:.0f}% {'>' if direction == 'UP' else '<' if direction == 'DOWN' else 'in'} "
            f"[{down_thresh:.0f},{up_thresh:.0f}]\n→{direction}",
            "active" if direction != "HOLD" else "blocked"
        )

        if direction == "HOLD":
            for i in range(2, 9):
                self._algo_blocks[i].update("—", "skipped")
        else:
            raw_desired = math.ceil(prev_replicas * smoothed / target)
            if direction == "UP":
                raw_desired = math.ceil(raw_desired * (1 + self._safety_var.get()))
            self._algo_blocks[2].update(f"{prev_replicas}×{smoothed:.0f}/{target:.0f}\n={raw_desired}", "active")
            clamped = max(2, min(8, raw_desired))
            self._algo_blocks[3].update(f"[2,8]→{clamped}", "active")
            rate_lim = min(clamped, prev_replicas + 2) if direction == "UP" else max(clamped, prev_replicas - 1)
            self._algo_blocks[4].update(f"±limit→{rate_lim}", "active")
            self._algo_blocks[5].update("OFF ✓", "active")
            self._algo_blocks[6].update("N/A" if direction == "UP" else "check", "active")

            executed = ml_replicas != prev_replicas
            self._algo_blocks[7].update(
                f"{ml_replicas}{'≠' if executed else '='}{prev_replicas}",
                "active" if executed else "blocked"
            )
            if executed:
                self._algo_blocks[8].update(f"{prev_replicas}→{ml_replicas}", "executed")
            else:
                self._algo_blocks[8].update("hold", "blocked")

        # State panel
        sim = self._ml_sim
        self._state_lbl.configure(
            text=f"Direction: {direction} | "
                 f"Consec. Below: {sim._consecutive_below} | "
                 f"Replicas: {ml_replicas} | "
                 f"Smoothed: {smoothed:.1f}%"
        )

        # CPU chart
        self._algo_cpu_ax.clear()
        if n > 1:
            x = range(n)
            self._algo_cpu_ax.plot(x, self._smoothed_trace, color=C["blue"], lw=1.5, label="Smoothed")
            self._algo_cpu_ax.plot(x, self._actual_trace, color=C["green"], lw=1, alpha=0.5, label="Actual")
            self._algo_cpu_ax.axhline(up_thresh, color=C["red"], ls="--", lw=0.8, label=f"Up {up_thresh:.0f}%")
            self._algo_cpu_ax.axhline(down_thresh, color=C["orange"], ls="--", lw=0.8, label=f"Down {down_thresh:.0f}%")
            self._algo_cpu_ax.axhline(target, color=C["yellow"], ls=":", lw=0.8, alpha=0.6)
            self._algo_cpu_ax.axhspan(down_thresh, up_thresh, alpha=0.08, color=C["yellow"])
            self._algo_cpu_ax.set_ylim(0, 100)
            self._algo_cpu_ax.set_ylabel("CPU %")
            self._algo_cpu_ax.legend(loc="upper right", fontsize=7)
            self._algo_cpu_ax.grid(True, alpha=0.2)
        self._algo_cpu_ax.set_title("CPU + Thresholds", fontsize=10)
        self._algo_cpu_fig.tight_layout()
        self._algo_cpu_canvas.draw_idle()

        # Replica chart
        self._algo_rep_ax.clear()
        if n > 1:
            x = range(n)
            self._algo_rep_ax.step(x, self._ml_replicas_trace, color=C["blue"], lw=2, where="post", label="ML Replicas")
            self._algo_rep_ax.step(x, self._hpa_replicas_trace, color=C["red"], lw=1.5, where="post", alpha=0.6, label="HPA Replicas")
            self._algo_rep_ax.axhline(2, color=C["text_dim"], ls=":", lw=0.7, alpha=0.5)
            self._algo_rep_ax.axhline(8, color=C["text_dim"], ls=":", lw=0.7, alpha=0.5)
            self._algo_rep_ax.set_ylim(0, 10)
            self._algo_rep_ax.set_ylabel("Replicas")
            self._algo_rep_ax.legend(loc="upper right", fontsize=7)
            self._algo_rep_ax.grid(True, alpha=0.2)
        self._algo_rep_ax.set_title("Replica Count", fontsize=10)
        self._algo_rep_fig.tight_layout()
        self._algo_rep_canvas.draw_idle()

        # Decision log
        self._decision_log.configure(state="normal")
        self._decision_log.delete("1.0", "end")
        # Show last 20 entries
        for line in self._decisions_log[-20:]:
            self._decision_log.insert("end", line + "\n")
        self._decision_log.configure(state="disabled")
        self._decision_log.see("end")

    def _update_exp_tab(self):
        if not hasattr(self, "_full_metrics"):
            return
        metrics = self._full_metrics
        history = self._history

        # Top chart: Actual CPU + replicas
        ax1, ax2 = self._exp_axes
        ax1.clear()
        ax2.clear()

        n = len(history)
        if n > 0:
            x = range(n)
            actual = [r.actual_cpu for r in history]
            hpa_r = [r.hpa_replicas for r in history]
            ml_r = [r.ml_replicas for r in history]

            # CPU trace (background)
            ax1.fill_between(x, actual, alpha=0.15, color=C["text_dim"])
            ax1.plot(x, actual, color=C["text_dim"], lw=0.8, alpha=0.5, label="Actual CPU")

            # Replicas
            ax1.step(x, [r * 10 for r in hpa_r], color=C["red"], lw=1.5,
                     where="post", label="HPA Replicas (×10)")
            ax1.step(x, [r * 10 for r in ml_r], color=C["blue"], lw=2,
                     where="post", label="ML Replicas (×10)")

            # SLA threshold
            target = self._target_var.get()
            ax1.axhline(target * 1.2, color=C["red"], ls=":", lw=0.8, alpha=0.5, label="SLA Limit")

            ax1.set_ylabel("CPU % / Replicas×10")
            ax1.set_ylim(0, 100)
            ax1.legend(loc="upper right", fontsize=7, ncol=2)
            ax1.set_title(f"Experiment: {self._scenarios[self._current_scenario_idx].name}", fontsize=11)
            ax1.grid(True, alpha=0.2)

            # Bottom chart: replica diff
            diff = [m - h for m, h in zip(ml_r, hpa_r)]
            colors_bar = [C["green"] if d < 0 else C["red"] if d > 0 else C["text_dim"] for d in diff]
            ax2.bar(x, diff, color=colors_bar, alpha=0.7, width=1.0)
            ax2.axhline(0, color=C["text_dim"], lw=0.5)
            ax2.set_ylabel("ML - HPA")
            ax2.set_xlabel("Step")
            ax2.grid(True, alpha=0.2)
            ax2.set_title("Replica Difference (negative = ML uses fewer)", fontsize=9)

        self._exp_fig.tight_layout()
        self._exp_canvas.draw_idle()

        # KPI cards
        self._kpi_cards[0].set_values(f"{metrics.hpa_avg_replicas:.1f}", f"{metrics.ml_avg_replicas:.1f}")
        self._kpi_cards[1].set_values(f"{metrics.hpa_replica_minutes:.0f}", f"{metrics.ml_replica_minutes:.0f}")
        self._kpi_cards[2].set_values(f"{metrics.hpa_sla_violations}", f"{metrics.ml_sla_violations}")
        self._kpi_cards[3].set_values(f"{metrics.hpa_scaling_events}", f"{metrics.ml_scaling_events}")
        self._kpi_cards[4].set_values(f"{metrics.hpa_over_provisioning_pct:.1f}%", f"{metrics.ml_over_provisioning_pct:.1f}%")
        self._kpi_cards[5].set_values(f"{metrics.hpa_under_provisioning_pct:.1f}%", f"{metrics.ml_under_provisioning_pct:.1f}%")
        self._kpi_cards[6].set_values(f"{metrics.hpa_avg_reaction_time:.1f}", f"{metrics.ml_avg_reaction_time:.1f}")
        self._kpi_cards[7].set_values(f"—", f"{metrics.cost_savings_pct:+.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
