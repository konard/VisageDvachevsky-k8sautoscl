"""ML Autoscaler — Experiment Dashboard (CustomTkinter).

Interactive visualization of:
  - Neural network internals (buffer, features, ONNX inference)
  - Scaling algorithm 9-step pipeline
  - HPA vs ML autoscaler comparison

Features:
  - Keyboard shortcuts (Space=play/pause, Right=step, R=reset, 1-4=scenario)
  - Tooltips on pipeline blocks and KPI cards
  - Live prediction accuracy metrics (MAE, RMSE)
  - Animated status indicators
  - Scenario description panel
  - Timeline progress bar
  - Export charts / data to file
  - Notification feed for scaling events
  - Configurable min/max replica range

Usage:
    python scripts/experiment_gui.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.experiment_scenarios import Scenario, get_all_scenarios  # noqa: E402
from scripts.experiment_simulator import (  # noqa: E402
    ExperimentRunner,
    MLAutoscalerSimulator,
    MLConfig,
    StepRecord,
)

# ---------------------------------------------------------------------------
# Color palette (unified with Grafana dashboards)
# ---------------------------------------------------------------------------

C = {
    "bg": "#1a1a2e",
    "bg_card": "#16213e",
    "sidebar": "#0f3460",
    "accent": "#e94560",
    "green": "#73BF69",
    "yellow": "#FADE2A",
    "orange": "#FF9830",
    "red": "#F2495C",
    "blue": "#5794F2",
    "purple": "#B877D9",
    "cyan": "#8AB8FF",
    "text": "#e0e0e0",
    "text_dim": "#8892b0",
    "grid": "#2a2a4a",
    "success": "#73BF69",
    "warning": "#FADE2A",
    "danger": "#F2495C",
}

# Matplotlib dark style
plt.rcParams.update(
    {
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
    }
)


# ---------------------------------------------------------------------------
# Utility: embed a matplotlib figure in a CTkFrame
# ---------------------------------------------------------------------------


def embed_figure(parent: ctk.CTkFrame, fig: Figure) -> FigureCanvasTkAgg:
    """Embed a matplotlib Figure into a CustomTkinter frame."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# ---------------------------------------------------------------------------
# Tooltip widget
# ---------------------------------------------------------------------------


class Tooltip:
    """Hover tooltip for any widget."""

    def __init__(self, widget: ctk.CTkBaseClass, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip_window: ctk.CTkToplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event: object = None) -> None:
        if self._tip_window:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        tw = ctk.CTkToplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        lbl = ctk.CTkLabel(
            tw,
            text=self._text,
            font=("Consolas", 10),
            fg_color=C["bg_card"],
            text_color=C["text"],
            corner_radius=6,
            wraplength=280,
        )
        lbl.pack(padx=6, pady=4)
        self._tip_window = tw

    def _hide(self, event: object = None) -> None:
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


# ---------------------------------------------------------------------------
# KPI Card widget (enhanced with delta indicator)
# ---------------------------------------------------------------------------


class KPICard(ctk.CTkFrame):
    """Compact KPI card: title + HPA value + ML value + delta arrow."""

    def __init__(self, master: ctk.CTkFrame, title: str, tooltip: str = "", **kw: object):
        super().__init__(master, fg_color=C["bg_card"], corner_radius=8, **kw)
        self._title_lbl = ctk.CTkLabel(
            self, text=title, font=("Consolas", 11), text_color=C["text_dim"]
        )
        self._title_lbl.pack(pady=(6, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=(0, 2))
        self._hpa_lbl = ctk.CTkLabel(
            row, text="HPA: --", font=("Consolas", 13, "bold"), text_color=C["red"]
        )
        self._hpa_lbl.pack(side="left", padx=6)
        self._ml_lbl = ctk.CTkLabel(
            row, text="ML: --", font=("Consolas", 13, "bold"), text_color=C["blue"]
        )
        self._ml_lbl.pack(side="left", padx=6)

        # Delta indicator
        self._delta_lbl = ctk.CTkLabel(
            self, text="", font=("Consolas", 9), text_color=C["text_dim"]
        )
        self._delta_lbl.pack(pady=(0, 4))

        if tooltip:
            Tooltip(self, tooltip)

    def set_values(
        self,
        hpa_val: str,
        ml_val: str,
        better: str = "lower",
    ) -> None:
        """Update card values with automatic delta comparison.

        Args:
            hpa_val: HPA metric value string.
            ml_val: ML metric value string.
            better: Direction that is "better" — "lower" or "higher".
        """
        self._hpa_lbl.configure(text=f"HPA: {hpa_val}")
        self._ml_lbl.configure(text=f"ML: {ml_val}")

        # Parse numeric values for delta
        try:
            hpa_num = float(hpa_val.replace("%", "").replace("+", "").replace(",", ""))
            ml_num = float(ml_val.replace("%", "").replace("+", "").replace(",", ""))
            diff = ml_num - hpa_num
            if abs(diff) < 0.01:
                self._delta_lbl.configure(text="= equal", text_color=C["text_dim"])
            else:
                is_better = (diff < 0) if better == "lower" else (diff > 0)
                arrow = "v" if diff < 0 else "^"
                color = C["green"] if is_better else C["red"]
                self._delta_lbl.configure(text=f"{arrow} {abs(diff):.1f}", text_color=color)
        except (ValueError, AttributeError):
            self._delta_lbl.configure(text="", text_color=C["text_dim"])


# ---------------------------------------------------------------------------
# Pipeline Step block (enhanced with tooltip)
# ---------------------------------------------------------------------------


class PipelineBlock(ctk.CTkFrame):
    """One step in the 9-step scaling algorithm visualization."""

    # Descriptions for each pipeline step
    STEP_TOOLTIPS = {
        1: "EMA: Exponential moving average smoothing. " "Smoothes predicted CPU to remove noise.",
        2: "Hysteresis: Dead-zone check. " "Determines UP/DOWN/HOLD based on thresholds.",
        3: "Desired: Calculates target replica count " "using ceil(current * smoothed / target).",
        4: "Clamp: Restricts replicas to [min, max] range.",
        5: "Rate Limit: Limits change speed. " "Max +2 up, -1 down per step.",
        6: "Cooldown: Prevents rapid re-scaling. " "60s for up, 300s for down.",
        7: "Stabilize: Requires 3 consecutive low " "readings before scaling down.",
        8: "No Change: Skips execution if desired " "equals current replicas.",
        9: "Commit: Applies the scaling decision " "and updates cooldown timers.",
    }

    def __init__(self, master: ctk.CTkFrame, step_num: int, title: str, **kw: object):
        super().__init__(
            master,
            fg_color=C["bg_card"],
            corner_radius=6,
            border_width=1,
            border_color=C["grid"],
            **kw,
        )
        self.configure(width=115, height=80)
        self.pack_propagate(False)
        self._step_num = step_num
        ctk.CTkLabel(
            self,
            text=f"Step {step_num}",
            font=("Consolas", 8),
            text_color=C["text_dim"],
        ).pack(pady=(4, 0))
        ctk.CTkLabel(
            self,
            text=title,
            font=("Consolas", 9, "bold"),
            text_color=C["text"],
        ).pack()
        self._value_lbl = ctk.CTkLabel(
            self,
            text="\u2014",
            font=("Consolas", 10),
            text_color=C["text_dim"],
            wraplength=100,
        )
        self._value_lbl.pack(pady=(2, 4))

        # Add tooltip
        tooltip_text = self.STEP_TOOLTIPS.get(step_num, "")
        if tooltip_text:
            Tooltip(self, tooltip_text)

    def update(self, text: str, status: str = "active") -> None:
        """Update block display text and status color."""
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
# Status Indicator (animated dot)
# ---------------------------------------------------------------------------


class StatusDot(ctk.CTkFrame):
    """Animated status dot indicator."""

    def __init__(self, master: ctk.CTkFrame, **kw: object):
        super().__init__(master, fg_color="transparent", **kw)
        self._dot = ctk.CTkLabel(
            self, text="\u25cf", font=("Consolas", 14), text_color=C["text_dim"]
        )
        self._dot.pack(side="left", padx=2)
        self._label = ctk.CTkLabel(
            self, text="IDLE", font=("Consolas", 10), text_color=C["text_dim"]
        )
        self._label.pack(side="left", padx=2)
        self._pulse_on = True

    def set_status(self, status: str, text: str) -> None:
        """Update status indicator.

        Args:
            status: One of 'idle', 'running', 'ready', 'warning'.
            text: Label text to display.
        """
        color_map = {
            "idle": C["text_dim"],
            "running": C["green"],
            "ready": C["blue"],
            "warning": C["yellow"],
            "error": C["red"],
        }
        color = color_map.get(status, C["text_dim"])
        self._dot.configure(text_color=color)
        self._label.configure(text=text, text_color=color)


# ---------------------------------------------------------------------------
# Notification Item
# ---------------------------------------------------------------------------


class NotificationItem(ctk.CTkFrame):
    """Single notification entry in the feed."""

    def __init__(
        self,
        master: ctk.CTkFrame,
        text: str,
        level: str = "info",
        **kw: object,
    ):
        super().__init__(master, fg_color=C["bg_card"], corner_radius=4, height=24, **kw)
        self.pack_propagate(False)
        color = {
            "info": C["blue"],
            "up": C["green"],
            "down": C["orange"],
            "warning": C["yellow"],
        }.get(level, C["text_dim"])
        icon = {
            "info": "\u25b6",
            "up": "\u25b2",
            "down": "\u25bc",
            "warning": "\u26a0",
        }.get(level, "\u25cf")
        ctk.CTkLabel(
            self,
            text=f" {icon} {text}",
            font=("Consolas", 9),
            text_color=color,
            anchor="w",
        ).pack(fill="x", padx=4, pady=2)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class ExperimentApp(ctk.CTk):
    """Main GUI application for the ML Autoscaler experiment dashboard."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ML Autoscaler \u2014 Experiment Dashboard")
        self.geometry("1480x960")
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

        # Notifications queue
        self._notifications: list[tuple[str, str]] = []

        # Build UI
        self._build_controls()
        self._build_sidebar()
        self._build_tabview()

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Load data
        self._load_scenarios()

    # -------------------------------------------------------------------
    # Keyboard shortcuts
    # -------------------------------------------------------------------

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts for common actions."""
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<Right>", lambda e: self._do_step())
        self.bind("<r>", lambda e: self._do_reset())
        self.bind("<R>", lambda e: self._do_reset())
        self.bind("<1>", lambda e: self._select_scenario(0))
        self.bind("<2>", lambda e: self._select_scenario(1))
        self.bind("<3>", lambda e: self._select_scenario(2))
        self.bind("<4>", lambda e: self._select_scenario(3))
        self.bind("<Control-e>", lambda e: self._export_data())
        self.bind("<Control-s>", lambda e: self._export_charts())

    # -------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------

    def _load_scenarios(self) -> None:
        try:
            self._scenarios = get_all_scenarios(
                ROOT / "data" / "processed" / "test.csv",
                ROOT / "data" / "processed" / "scaler.pkl",
                ROOT / "data" / "processed" / "metadata.json",
            )
        except Exception:
            self._scenarios = [self._synthetic_fallback()]
        # Ensure we have at least 4 scenario buttons
        while len(self._scenarios) < 4:
            self._scenarios.append(self._synthetic_fallback())
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
            description="Synthetic sine wave demo for testing",
            actual_cpu=np.clip(actual, 10, 100),
            predicted_cpu=np.clip(predicted, 10, 100),
            duration_hours=n * 5 / 60,
        )

    def _select_scenario(self, idx: int) -> None:
        if idx >= len(self._scenarios):
            return
        self._current_scenario_idx = idx
        self._reset_simulation()
        self._update_all()

    def _do_reset(self) -> None:
        """Reset simulation (callable from shortcuts)."""
        self._reset_simulation()
        self._update_all()

    def _reset_simulation(self) -> None:
        self._step = 0
        self._playing = False
        self._notifications = []
        sc = self._scenarios[self._current_scenario_idx]

        # Run full experiment for Tab 3
        runner = ExperimentRunner(
            sc.actual_cpu,
            sc.predicted_cpu,
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

        # Update play button state
        if hasattr(self, "_play_btn"):
            self._play_btn.configure(text="\u25b6  Play", fg_color=C["green"])

    def _build_ml_config(self) -> MLConfig:
        return MLConfig(
            target_utilization=self._target_var.get(),
            smoothing_alpha=self._alpha_var.get(),
            safety_margin=self._safety_var.get(),
            scale_up_threshold_factor=self._up_thresh_var.get(),
            scale_down_threshold_factor=self._down_thresh_var.get(),
            min_replicas=int(self._min_rep_var.get()),
            max_replicas=int(self._max_rep_var.get()),
        )

    # -------------------------------------------------------------------
    # Control bar (top)
    # -------------------------------------------------------------------

    def _build_controls(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=C["bg_card"], height=50)
        bar.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            bar,
            text="ML Autoscaler \u2014 Experiment Dashboard",
            font=("Consolas", 16, "bold"),
            text_color=C["text"],
        ).pack(side="left", padx=12)

        # Status dot
        self._status_dot = StatusDot(bar)
        self._status_dot.pack(side="left", padx=8)

        # Transport controls
        self._play_btn = ctk.CTkButton(
            bar,
            text="\u25b6  Play",
            width=90,
            fg_color=C["green"],
            text_color="#000",
            hover_color="#5ea554",
            command=self._toggle_play,
        )
        self._play_btn.pack(side="left", padx=4)

        ctk.CTkButton(
            bar,
            text="\u23ed  Step",
            width=90,
            fg_color=C["blue"],
            text_color="#000",
            hover_color="#4678c9",
            command=self._do_step,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            bar,
            text="\u21ba  Reset",
            width=90,
            fg_color=C["orange"],
            text_color="#000",
            hover_color="#d97e24",
            command=self._do_reset,
        ).pack(side="left", padx=4)

        # Export buttons
        ctk.CTkButton(
            bar,
            text="\u2913 Export",
            width=80,
            fg_color=C["purple"],
            text_color="#fff",
            hover_color="#9b5fbf",
            command=self._export_data,
        ).pack(side="left", padx=4)

        # Step counter + time display
        self._time_lbl = ctk.CTkLabel(
            bar,
            text="00:00",
            font=("Consolas", 11),
            text_color=C["text_dim"],
        )
        self._time_lbl.pack(side="right", padx=4)

        self._step_lbl = ctk.CTkLabel(
            bar,
            text="Step: 0 / 0",
            font=("Consolas", 12),
            text_color=C["text_dim"],
        )
        self._step_lbl.pack(side="right", padx=8)

        # Speed slider
        ctk.CTkLabel(bar, text="Speed:", text_color=C["text_dim"]).pack(side="right")
        self._speed_slider = ctk.CTkSlider(
            bar,
            from_=50,
            to=1000,
            width=120,
            command=lambda v: setattr(self, "_speed_ms", int(v)),
        )
        self._speed_slider.set(200)
        self._speed_slider.pack(side="right", padx=4)

        # Shortcuts hint
        Tooltip(
            self._play_btn,
            "Space: Play/Pause | Right: Step | R: Reset | 1-4: Scenario | "
            "Ctrl+E: Export data | Ctrl+S: Save charts",
        )

    # -------------------------------------------------------------------
    # Timeline progress bar
    # -------------------------------------------------------------------

    def _build_timeline(self) -> None:
        """Build timeline progress bar below tabs."""
        frame = ctk.CTkFrame(self, fg_color=C["bg_card"], height=30)
        frame.pack(fill="x", padx=8, pady=(0, 2))
        frame.pack_propagate(False)

        self._timeline_bar = ctk.CTkProgressBar(
            frame, width=600, height=8, progress_color=C["blue"]
        )
        self._timeline_bar.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        self._timeline_bar.set(0)

        self._timeline_pct = ctk.CTkLabel(
            frame, text="0%", font=("Consolas", 9), text_color=C["text_dim"]
        )
        self._timeline_pct.pack(side="right", padx=8)

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._play_btn.configure(
            text="\u23f8  Pause" if self._playing else "\u25b6  Play",
            fg_color=C["yellow"] if self._playing else C["green"],
        )
        self._status_dot.set_status(
            "running" if self._playing else "idle",
            "RUNNING" if self._playing else "PAUSED",
        )
        if self._playing:
            self._auto_step()

    def _auto_step(self) -> None:
        if not self._playing:
            return
        sc = self._scenarios[self._current_scenario_idx]
        if self._step < len(sc.actual_cpu):
            self._do_step()
            self.after(self._speed_ms, self._auto_step)
        else:
            self._playing = False
            self._play_btn.configure(text="\u25b6  Play", fg_color=C["green"])
            self._status_dot.set_status("ready", "FINISHED")

    def _do_step(self) -> None:
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
            f"[t={t:>3}] {ml_dir:>4} | ML:{ml_replicas} "
            f"HPA:{hpa_replicas} | "
            f"actual={actual:.1f}% pred={predicted:.1f}% "
            f"smooth={smoothed:.1f}%"
        )

        # Generate notification for scaling events
        if ml_dir == "up":
            prev = self._ml_replicas_trace[-2] if len(self._ml_replicas_trace) > 1 else 2
            self._add_notification(f"t={t}: ML SCALE UP {prev} -> {ml_replicas}", "up")
        elif ml_dir == "down":
            prev = self._ml_replicas_trace[-2] if len(self._ml_replicas_trace) > 1 else 2
            self._add_notification(f"t={t}: ML SCALE DOWN {prev} -> {ml_replicas}", "down")

        # Warn if CPU is high
        if actual > 85:
            self._add_notification(f"t={t}: High CPU alert: {actual:.1f}%", "warning")

        self._step += 1
        self._update_all()

    # -------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------

    def _add_notification(self, text: str, level: str = "info") -> None:
        """Add a notification to the feed (max 50 kept)."""
        self._notifications.append((text, level))
        if len(self._notifications) > 50:
            self._notifications = self._notifications[-50:]

    # -------------------------------------------------------------------
    # Export functionality
    # -------------------------------------------------------------------

    def _export_data(self) -> None:
        """Export simulation data to CSV file."""
        if not self._actual_trace:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export simulation data",
        )
        if not filepath:
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "actual_cpu",
                    "predicted_cpu",
                    "smoothed_cpu",
                    "ml_replicas",
                    "hpa_replicas",
                ]
            )
            for i in range(len(self._actual_trace)):
                writer.writerow(
                    [
                        i,
                        f"{self._actual_trace[i]:.2f}",
                        f"{self._predicted_trace[i]:.2f}",
                        f"{self._smoothed_trace[i]:.2f}",
                        self._ml_replicas_trace[i],
                        self._hpa_replicas_trace[i],
                    ]
                )
        self._add_notification(f"Data exported to {filepath}", "info")

    def _export_charts(self) -> None:
        """Export current charts as PNG images."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
            title="Save charts",
        )
        if not filepath:
            return
        # Save all figures
        base = Path(filepath).stem
        parent = Path(filepath).parent
        figures = {
            "nn": self._nn_fig,
            "algo_cpu": self._algo_cpu_fig,
            "algo_rep": self._algo_rep_fig,
            "experiment": self._exp_fig,
        }
        for name, fig in figures.items():
            out = parent / f"{base}_{name}.png"
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
        self._add_notification(f"Charts saved to {parent}/{base}_*.png", "info")

    # -------------------------------------------------------------------
    # Sidebar (parameters)
    # -------------------------------------------------------------------

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkScrollableFrame(self, fg_color=C["sidebar"], width=220, corner_radius=0)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)

        ctk.CTkLabel(
            sidebar,
            text="Parameters",
            font=("Consolas", 14, "bold"),
            text_color=C["text"],
        ).pack(pady=(12, 8))

        def make_slider(
            parent: ctk.CTkFrame,
            label: str,
            from_: float,
            to: float,
            default: float,
        ) -> ctk.DoubleVar:
            ctk.CTkLabel(
                parent,
                text=label,
                font=("Consolas", 10),
                text_color=C["text_dim"],
            ).pack(pady=(8, 0), padx=8, anchor="w")
            var = ctk.DoubleVar(value=default)
            val_lbl = ctk.CTkLabel(
                parent,
                text=f"{default}",
                font=("Consolas", 10),
                text_color=C["text"],
            )
            val_lbl.pack(padx=8, anchor="e")

            def on_change(v: str) -> None:
                val_lbl.configure(text=f"{float(v):.2f}")

            slider = ctk.CTkSlider(
                parent,
                from_=from_,
                to=to,
                variable=var,
                width=180,
                command=on_change,
            )
            slider.pack(padx=8)
            return var

        self._target_var = make_slider(sidebar, "Target Util %", 20, 90, 60)
        self._alpha_var = make_slider(sidebar, "EMA Alpha", 0.1, 1.0, 0.7)
        self._safety_var = make_slider(sidebar, "Safety Margin", 0.0, 0.5, 0.15)
        self._up_thresh_var = make_slider(sidebar, "Up Threshold \u00d7", 1.0, 1.5, 1.1)
        self._down_thresh_var = make_slider(sidebar, "Down Threshold \u00d7", 0.3, 0.9, 0.7)

        # Replica range sliders
        ctk.CTkLabel(
            sidebar,
            text="\u2500\u2500\u2500 Replica Range \u2500\u2500\u2500",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        ).pack(pady=(12, 0), padx=8)
        self._min_rep_var = make_slider(sidebar, "Min Replicas", 1, 5, 2)
        self._max_rep_var = make_slider(sidebar, "Max Replicas", 4, 20, 8)

        ctk.CTkButton(
            sidebar,
            text="Apply & Re-run",
            fg_color=C["accent"],
            text_color="#fff",
            hover_color="#c73550",
            command=self._do_reset,
        ).pack(pady=16, padx=12, fill="x")

        # Scenario selector
        ctk.CTkLabel(
            sidebar,
            text="Scenario",
            font=("Consolas", 14, "bold"),
            text_color=C["text"],
        ).pack(pady=(16, 4))

        scenario_info = [
            ("Stable", "Low-variance workload (~45-55%)"),
            ("Ramp", "30% -> 85% gradual increase"),
            ("Spike", "3 sudden load spikes"),
            ("Diurnal", "48h day/night cycle"),
        ]
        self._scenario_btns: list[ctk.CTkButton] = []
        for i, (name, desc) in enumerate(scenario_info):
            btn = ctk.CTkButton(
                sidebar,
                text=name,
                width=180,
                height=28,
                fg_color=C["bg_card"],
                hover_color=C["blue"],
                command=lambda idx=i: self._select_scenario(idx),
            )
            btn.pack(pady=2, padx=12)
            Tooltip(btn, desc)
            self._scenario_btns.append(btn)

        # Scenario description
        self._scenario_desc = ctk.CTkLabel(
            sidebar,
            text="",
            font=("Consolas", 9),
            text_color=C["text_dim"],
            wraplength=200,
        )
        self._scenario_desc.pack(pady=(8, 4), padx=8)

        # Keyboard shortcuts help
        ctk.CTkLabel(
            sidebar,
            text="\u2500\u2500\u2500 Shortcuts \u2500\u2500\u2500",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        ).pack(pady=(16, 4), padx=8)
        shortcuts = [
            "Space: Play/Pause",
            "Right Arrow: Step",
            "R: Reset",
            "1-4: Select Scenario",
            "Ctrl+E: Export Data",
            "Ctrl+S: Save Charts",
        ]
        for sc in shortcuts:
            ctk.CTkLabel(
                sidebar,
                text=sc,
                font=("Consolas", 8),
                text_color=C["text_dim"],
            ).pack(padx=12, anchor="w")

    # -------------------------------------------------------------------
    # Tab view (4 tabs)
    # -------------------------------------------------------------------

    def _build_tabview(self) -> None:
        self._tabview = ctk.CTkTabview(self, fg_color=C["bg"])
        self._tabview.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self._tab_nn = self._tabview.add("Neural Network")
        self._tab_algo = self._tabview.add("Scaling Algorithm")
        self._tab_exp = self._tabview.add("Experiment")
        self._tab_log = self._tabview.add("Activity Log")

        self._build_nn_tab()
        self._build_algo_tab()
        self._build_exp_tab()
        self._build_log_tab()

        # Timeline at the bottom
        self._build_timeline()

    # --- Tab 1: Neural Network ---

    def _build_nn_tab(self) -> None:
        tab = self._tab_nn

        # Top: pipeline diagram
        pipe_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8, height=90)
        pipe_frame.pack(fill="x", padx=8, pady=4)
        pipe_frame.pack_propagate(False)

        ctk.CTkLabel(
            pipe_frame,
            text="ONNX Inference Pipeline",
            font=("Consolas", 12, "bold"),
            text_color=C["text"],
        ).pack(pady=(6, 4))

        blocks_frame = ctk.CTkFrame(pipe_frame, fg_color="transparent")
        blocks_frame.pack()
        stages = [
            ("Raw Metrics", "5 values", C["green"]),
            ("\u2192", "", C["text_dim"]),
            ("Feature Eng.", "44 features", C["purple"]),
            ("\u2192", "", C["text_dim"]),
            ("StandardScaler", "normalize", C["orange"]),
            ("\u2192", "", C["text_dim"]),
            ("ONNX LSTM", "(1,24,44)\u2192(1,1)", C["blue"]),
            ("\u2192", "", C["text_dim"]),
            ("Inverse Scale", "\u2192 CPU %", C["green"]),
        ]
        stage_tooltips = [
            "5 raw metrics: cpu_usage, mem_util, net_in, net_out, disk_io",
            "",
            "Generates 44 features: time features, lags, rolling windows",
            "",
            "Normalizes features using saved StandardScaler (mean/std)",
            "",
            "2-layer LSTM, 65K params, 0.13ms latency via ONNX Runtime",
            "",
            "Converts normalized prediction back to CPU percentage",
        ]
        for i, (title, sub, color) in enumerate(stages):
            if title == "\u2192":
                ctk.CTkLabel(
                    blocks_frame,
                    text="\u2192",
                    font=("Consolas", 16, "bold"),
                    text_color=C["text_dim"],
                ).pack(side="left", padx=2)
            else:
                f = ctk.CTkFrame(
                    blocks_frame,
                    fg_color=C["bg"],
                    corner_radius=6,
                    border_width=1,
                    border_color=color,
                )
                f.pack(side="left", padx=4)
                ctk.CTkLabel(
                    f,
                    text=title,
                    font=("Consolas", 9, "bold"),
                    text_color=color,
                ).pack(padx=6, pady=(2, 0))
                ctk.CTkLabel(
                    f,
                    text=sub,
                    font=("Consolas", 8),
                    text_color=C["text_dim"],
                ).pack(padx=6, pady=(0, 2))
                if stage_tooltips[i]:
                    Tooltip(f, stage_tooltips[i])

        # Middle: prediction output + buffer + accuracy
        mid = ctk.CTkFrame(tab, fg_color="transparent")
        mid.pack(fill="x", padx=8, pady=4)

        # Buffer status
        buf_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8, width=220)
        buf_frame.pack(side="left", padx=(0, 8), fill="y")
        buf_frame.pack_propagate(False)
        ctk.CTkLabel(
            buf_frame,
            text="Buffer Status",
            font=("Consolas", 11, "bold"),
            text_color=C["text"],
        ).pack(pady=(8, 4))
        self._buf_progress = ctk.CTkProgressBar(buf_frame, width=180)
        self._buf_progress.pack(pady=4, padx=12)
        self._buf_progress.set(0)
        self._buf_lbl = ctk.CTkLabel(
            buf_frame,
            text="0 / 24  NOT READY",
            font=("Consolas", 10),
            text_color=C["red"],
        )
        self._buf_lbl.pack()
        Tooltip(
            buf_frame,
            "LSTM requires 24 timesteps (2h of 5-min data) before predictions.",
        )

        # Prediction big number
        pred_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8, width=180)
        pred_frame.pack(side="left", padx=8, fill="y")
        pred_frame.pack_propagate(False)
        ctk.CTkLabel(
            pred_frame,
            text="ML Prediction",
            font=("Consolas", 11, "bold"),
            text_color=C["text"],
        ).pack(pady=(8, 2))
        self._pred_big = ctk.CTkLabel(
            pred_frame,
            text="\u2014",
            font=("Consolas", 28, "bold"),
            text_color=C["blue"],
        )
        self._pred_big.pack(pady=4)
        self._pred_actual = ctk.CTkLabel(
            pred_frame,
            text="Actual: \u2014",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        )
        self._pred_actual.pack()

        # Prediction accuracy panel
        acc_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8, width=200)
        acc_frame.pack(side="left", padx=8, fill="y")
        acc_frame.pack_propagate(False)
        ctk.CTkLabel(
            acc_frame,
            text="Prediction Accuracy",
            font=("Consolas", 11, "bold"),
            text_color=C["text"],
        ).pack(pady=(8, 4))
        self._acc_mae_lbl = ctk.CTkLabel(
            acc_frame,
            text="MAE: \u2014",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        )
        self._acc_mae_lbl.pack(anchor="w", padx=12)
        self._acc_rmse_lbl = ctk.CTkLabel(
            acc_frame,
            text="RMSE: \u2014",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        )
        self._acc_rmse_lbl.pack(anchor="w", padx=12)
        self._acc_err_lbl = ctk.CTkLabel(
            acc_frame,
            text="Error: \u2014",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        )
        self._acc_err_lbl.pack(anchor="w", padx=12)
        Tooltip(
            acc_frame,
            "MAE: Mean Absolute Error\n"
            "RMSE: Root Mean Squared Error\n"
            "Error: Current |predicted - actual|",
        )

        # Raw metrics panel
        raw_frame = ctk.CTkFrame(mid, fg_color=C["bg_card"], corner_radius=8)
        raw_frame.pack(side="left", fill="both", expand=True, padx=8)
        ctk.CTkLabel(
            raw_frame,
            text="Current Metrics",
            font=("Consolas", 11, "bold"),
            text_color=C["text"],
        ).pack(pady=(8, 4))
        self._raw_labels: dict[str, ctk.CTkLabel] = {}
        metrics_info = {
            "cpu_usage": "CPU utilization (%)",
            "mem_util": "Memory utilization (%)",
            "net_in": "Network inbound (KB/s)",
            "net_out": "Network outbound (KB/s)",
            "disk_io": "Disk I/O utilization (%)",
        }
        for metric, desc in metrics_info.items():
            row = ctk.CTkFrame(raw_frame, fg_color="transparent")
            row.pack(fill="x", padx=12)
            lbl = ctk.CTkLabel(
                row,
                text=f"{metric}: \u2014",
                font=("Consolas", 10),
                text_color=C["text_dim"],
            )
            lbl.pack(side="left")
            self._raw_labels[metric] = lbl
            # Mini progress bar for each metric
            bar = ctk.CTkProgressBar(row, width=60, height=6)
            bar.pack(side="right", padx=4)
            bar.set(0)
            self._raw_labels[f"{metric}_bar"] = bar
        Tooltip(raw_frame, "Simulated from CPU data. In production, " "these come from Prometheus.")

        # Bottom: prediction history chart
        self._nn_fig, self._nn_ax = plt.subplots(1, 1, figsize=(10, 3), dpi=80)
        self._nn_canvas = embed_figure(tab, self._nn_fig)

    # --- Tab 2: Scaling Algorithm ---

    def _build_algo_tab(self) -> None:
        tab = self._tab_algo

        # 9-step pipeline
        pipe = ctk.CTkFrame(tab, fg_color="transparent")
        pipe.pack(fill="x", padx=4, pady=4)

        step_names = [
            "EMA",
            "Hysteresis",
            "Desired",
            "Clamp",
            "Rate Limit",
            "Cooldown",
            "Stabilize",
            "No Change?",
            "Commit",
        ]
        self._algo_blocks: list[PipelineBlock] = []
        for i, name in enumerate(step_names):
            if i > 0:
                ctk.CTkLabel(
                    pipe,
                    text="\u2192",
                    font=("Consolas", 14, "bold"),
                    text_color=C["text_dim"],
                ).pack(side="left", padx=1)
            block = PipelineBlock(pipe, i + 1, name)
            block.pack(side="left", padx=2)
            self._algo_blocks.append(block)

        # State panel
        state_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8, height=40)
        state_frame.pack(fill="x", padx=8, pady=4)
        self._state_lbl = ctk.CTkLabel(
            state_frame,
            text=(
                "Cooldown UP: \u2014 | Cooldown DOWN: \u2014 | "
                "Consec. Below: \u2014 | Direction: \u2014"
            ),
            font=("Consolas", 10),
            text_color=C["text_dim"],
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
            tab,
            height=100,
            font=("Consolas", 9),
            fg_color=C["bg"],
            text_color=C["text_dim"],
        )
        self._decision_log.pack(fill="x", padx=8, pady=(0, 4))

    # --- Tab 3: Experiment Comparison ---

    def _build_exp_tab(self) -> None:
        tab = self._tab_exp

        # Main comparison chart
        self._exp_fig, self._exp_axes = plt.subplots(
            3,
            1,
            figsize=(10, 6),
            dpi=80,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        self._exp_canvas = embed_figure(tab, self._exp_fig)

        # KPI cards grid
        cards_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=8, pady=4)

        kpi_info = [
            ("Avg Replicas", "lower", "Average number of running replicas"),
            ("Replica-Min", "lower", "Total replica-minutes consumed"),
            (
                "SLA Violations",
                "lower",
                "Steps where CPU exceeded SLA threshold",
            ),
            (
                "Scaling Events",
                "lower",
                "Number of replica count changes",
            ),
            (
                "Over-Prov %",
                "lower",
                "Percentage of over-provisioned capacity",
            ),
            (
                "Under-Prov %",
                "lower",
                "Percentage of under-provisioned capacity",
            ),
            (
                "Reaction Time",
                "lower",
                "Avg steps to respond to demand spike",
            ),
            (
                "Cost Savings %",
                "higher",
                "ML cost savings vs HPA (negative = higher cost)",
            ),
        ]
        self._kpi_cards: list[KPICard] = []
        self._kpi_better: list[str] = []
        for i, (name, better, tooltip) in enumerate(kpi_info):
            card = KPICard(cards_frame, name, tooltip=tooltip)
            card.grid(row=i // 4, column=i % 4, padx=4, pady=4, sticky="nsew")
            self._kpi_cards.append(card)
            self._kpi_better.append(better)
        for c in range(4):
            cards_frame.columnconfigure(c, weight=1)

    # --- Tab 4: Activity Log ---

    def _build_log_tab(self) -> None:
        tab = self._tab_log

        ctk.CTkLabel(
            tab,
            text="Activity & Notification Feed",
            font=("Consolas", 14, "bold"),
            text_color=C["text"],
        ).pack(pady=(8, 4), padx=8, anchor="w")

        ctk.CTkLabel(
            tab,
            text="Scaling events, alerts, and system notifications",
            font=("Consolas", 10),
            text_color=C["text_dim"],
        ).pack(padx=8, anchor="w")

        self._notif_frame = ctk.CTkScrollableFrame(tab, fg_color=C["bg"], corner_radius=8)
        self._notif_frame.pack(fill="both", expand=True, padx=8, pady=8)

    # -------------------------------------------------------------------
    # Update all visuals
    # -------------------------------------------------------------------

    def _update_all(self) -> None:
        sc = self._scenarios[self._current_scenario_idx]
        total = len(sc.actual_cpu)
        self._step_lbl.configure(text=f"Step: {self._step} / {total}")

        # Time display (5 min per step)
        minutes = self._step * 5
        hours = minutes // 60
        mins = minutes % 60
        self._time_lbl.configure(text=f"{hours:02d}:{mins:02d}")

        # Timeline
        progress = self._step / total if total > 0 else 0
        self._timeline_bar.set(progress)
        self._timeline_pct.configure(text=f"{progress * 100:.0f}%")

        # Highlight active scenario button
        for i, btn in enumerate(self._scenario_btns):
            btn.configure(fg_color=C["blue"] if i == self._current_scenario_idx else C["bg_card"])

        # Update scenario description
        self._scenario_desc.configure(text=sc.description)

        self._update_nn_tab()
        self._update_algo_tab()
        self._update_exp_tab()
        self._update_log_tab()

    def _update_nn_tab(self) -> None:
        n = self._step

        # Buffer status
        buf_fill = min(n, 24)
        self._buf_progress.set(buf_fill / 24)
        ready = n >= 24
        self._buf_lbl.configure(
            text=f"{min(n, 24)} / 24  {'READY' if ready else 'FILLING...'}",
            text_color=C["green"] if ready else C["red"],
        )

        # Current prediction
        if n > 0:
            pred = self._predicted_trace[-1]
            actual = self._actual_trace[-1]
            color = (
                C["green"]
                if pred < 42
                else C["yellow"] if pred < 66 else C["orange"] if pred < 85 else C["red"]
            )
            self._pred_big.configure(text=f"{pred:.1f}%", text_color=color)
            self._pred_actual.configure(text=f"Actual: {actual:.1f}%")

            # Raw metrics (simulated from CPU)
            metrics_vals = {
                "cpu_usage": (actual, 100),
                "mem_util": (actual * 0.6, 100),
                "net_in": (actual * 1.2, 120),
                "net_out": (actual * 0.8, 80),
                "disk_io": (actual * 0.3, 30),
            }
            for metric, (val, max_val) in metrics_vals.items():
                unit = "%" if "usage" in metric or "util" in metric or "io" in metric else " KB/s"
                self._raw_labels[metric].configure(text=f"{metric}: {val:.1f}{unit}")
                bar_key = f"{metric}_bar"
                if bar_key in self._raw_labels:
                    bar_val = min(val / max_val, 1.0) if max_val > 0 else 0
                    self._raw_labels[bar_key].set(bar_val)

            # Prediction accuracy
            errors = np.abs(np.array(self._actual_trace) - np.array(self._predicted_trace))
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors**2)))
            cur_err = abs(actual - pred)

            mae_color = C["green"] if mae < 5 else C["yellow"] if mae < 10 else C["red"]
            self._acc_mae_lbl.configure(text=f"MAE: {mae:.2f}%", text_color=mae_color)
            self._acc_rmse_lbl.configure(text=f"RMSE: {rmse:.2f}%", text_color=mae_color)
            err_color = C["green"] if cur_err < 5 else C["yellow"] if cur_err < 10 else C["red"]
            self._acc_err_lbl.configure(text=f"Error: {cur_err:.2f}%", text_color=err_color)

        # Prediction history chart
        self._nn_ax.clear()
        if n > 1:
            window = 80
            x = list(range(max(0, n - window), n))
            a_slice = self._actual_trace[max(0, n - window) :]
            p_slice = self._predicted_trace[max(0, n - window) :]
            self._nn_ax.plot(x, a_slice, color=C["green"], lw=1.5, label="Actual CPU")
            self._nn_ax.plot(
                x,
                p_slice,
                color=C["purple"],
                lw=1.5,
                ls="--",
                label="Predicted",
            )
            # Fill error band
            self._nn_ax.fill_between(x, a_slice, p_slice, alpha=0.12, color=C["purple"])
            self._nn_ax.axhline(60, color=C["yellow"], ls=":", lw=0.8, alpha=0.5)
            self._nn_ax.set_ylabel("CPU %")
            self._nn_ax.set_ylim(0, 100)
            self._nn_ax.legend(loc="upper right", fontsize=8)
            self._nn_ax.grid(True, alpha=0.2)
        self._nn_ax.set_title(
            f"Prediction History (last 80 steps) | Step {n}",
            fontsize=10,
        )
        self._nn_fig.tight_layout()
        self._nn_canvas.draw_idle()

    def _update_algo_tab(self) -> None:
        n = self._step
        if n == 0:
            for block in self._algo_blocks:
                block.update("\u2014", "skipped")
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
        self._algo_blocks[0].update(f"{pred:.1f}\u2192{smoothed:.1f}%", "active")
        cmp_sym = ">" if direction == "UP" else "<" if direction == "DOWN" else "in"
        self._algo_blocks[1].update(
            f"{smoothed:.0f}% {cmp_sym} "
            f"[{down_thresh:.0f},{up_thresh:.0f}]"
            f"\n\u2192{direction}",
            "active" if direction != "HOLD" else "blocked",
        )

        min_r = int(self._min_rep_var.get())
        max_r = int(self._max_rep_var.get())

        if direction == "HOLD":
            for i in range(2, 9):
                self._algo_blocks[i].update("\u2014", "skipped")
        else:
            raw_desired = math.ceil(prev_replicas * smoothed / target)
            if direction == "UP":
                raw_desired = math.ceil(raw_desired * (1 + self._safety_var.get()))
            self._algo_blocks[2].update(
                f"{prev_replicas}\u00d7{smoothed:.0f}" f"/{target:.0f}\n={raw_desired}",
                "active",
            )
            clamped = max(min_r, min(max_r, raw_desired))
            self._algo_blocks[3].update(f"[{min_r},{max_r}]\u2192{clamped}", "active")
            rate_lim = (
                min(clamped, prev_replicas + 2)
                if direction == "UP"
                else max(clamped, prev_replicas - 1)
            )
            self._algo_blocks[4].update(f"\u00b1limit\u2192{rate_lim}", "active")
            self._algo_blocks[5].update("OFF \u2713", "active")
            self._algo_blocks[6].update("N/A" if direction == "UP" else "check", "active")

            executed = ml_replicas != prev_replicas
            eq_sym = "\u2260" if executed else "="
            self._algo_blocks[7].update(
                f"{ml_replicas}{eq_sym}{prev_replicas}",
                "active" if executed else "blocked",
            )
            if executed:
                self._algo_blocks[8].update(f"{prev_replicas}\u2192{ml_replicas}", "executed")
            else:
                self._algo_blocks[8].update("hold", "blocked")

        # State panel
        sim = self._ml_sim
        dir_color = {
            "UP": C["green"],
            "DOWN": C["orange"],
            "HOLD": C["text_dim"],
        }.get(direction, C["text_dim"])
        self._state_lbl.configure(
            text=f"Direction: {direction} | "
            f"Consec. Below: {sim._consecutive_below} | "
            f"Replicas: {ml_replicas} | "
            f"Smoothed: {smoothed:.1f}%",
            text_color=dir_color,
        )

        # CPU chart
        self._algo_cpu_ax.clear()
        if n > 1:
            x = list(range(n))
            self._algo_cpu_ax.plot(
                x,
                self._smoothed_trace,
                color=C["blue"],
                lw=1.5,
                label="Smoothed",
            )
            self._algo_cpu_ax.plot(
                x,
                self._actual_trace,
                color=C["green"],
                lw=1,
                alpha=0.5,
                label="Actual",
            )
            self._algo_cpu_ax.axhline(
                up_thresh,
                color=C["red"],
                ls="--",
                lw=0.8,
                label=f"Up {up_thresh:.0f}%",
            )
            self._algo_cpu_ax.axhline(
                down_thresh,
                color=C["orange"],
                ls="--",
                lw=0.8,
                label=f"Down {down_thresh:.0f}%",
            )
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
            x = list(range(n))
            self._algo_rep_ax.step(
                x,
                self._ml_replicas_trace,
                color=C["blue"],
                lw=2,
                where="post",
                label="ML Replicas",
            )
            self._algo_rep_ax.step(
                x,
                self._hpa_replicas_trace,
                color=C["red"],
                lw=1.5,
                where="post",
                alpha=0.6,
                label="HPA Replicas",
            )
            self._algo_rep_ax.axhline(
                min_r,
                color=C["text_dim"],
                ls=":",
                lw=0.7,
                alpha=0.5,
            )
            self._algo_rep_ax.axhline(
                max_r,
                color=C["text_dim"],
                ls=":",
                lw=0.7,
                alpha=0.5,
            )
            self._algo_rep_ax.set_ylim(0, max_r + 2)
            self._algo_rep_ax.set_ylabel("Replicas")
            self._algo_rep_ax.legend(loc="upper right", fontsize=7)
            self._algo_rep_ax.grid(True, alpha=0.2)
        self._algo_rep_ax.set_title("Replica Count", fontsize=10)
        self._algo_rep_fig.tight_layout()
        self._algo_rep_canvas.draw_idle()

        # Decision log
        self._decision_log.configure(state="normal")
        self._decision_log.delete("1.0", "end")
        for line in self._decisions_log[-20:]:
            self._decision_log.insert("end", line + "\n")
        self._decision_log.configure(state="disabled")
        self._decision_log.see("end")

    def _update_exp_tab(self) -> None:
        if not hasattr(self, "_full_metrics"):
            return
        metrics = self._full_metrics
        history = self._history

        ax1, ax2, ax3 = self._exp_axes
        ax1.clear()
        ax2.clear()
        ax3.clear()

        n = len(history)
        if n > 0:
            x = list(range(n))
            actual = [r.actual_cpu for r in history]
            hpa_r = [r.hpa_replicas for r in history]
            ml_r = [r.ml_replicas for r in history]

            # Chart 1: CPU + Replicas overlay
            ax1.fill_between(x, actual, alpha=0.15, color=C["text_dim"])
            ax1.plot(
                x,
                actual,
                color=C["text_dim"],
                lw=0.8,
                alpha=0.5,
                label="Actual CPU",
            )
            ax1.step(
                x,
                [r * 10 for r in hpa_r],
                color=C["red"],
                lw=1.5,
                where="post",
                label="HPA Replicas (\u00d710)",
            )
            ax1.step(
                x,
                [r * 10 for r in ml_r],
                color=C["blue"],
                lw=2,
                where="post",
                label="ML Replicas (\u00d710)",
            )
            target = self._target_var.get()
            ax1.axhline(
                target * 1.2,
                color=C["red"],
                ls=":",
                lw=0.8,
                alpha=0.5,
                label="SLA Limit",
            )
            ax1.set_ylabel("CPU % / Replicas\u00d710")
            ax1.set_ylim(0, 100)
            ax1.legend(loc="upper right", fontsize=7, ncol=2)
            sc = self._scenarios[self._current_scenario_idx]
            ax1.set_title(f"Experiment: {sc.name}", fontsize=11)
            ax1.grid(True, alpha=0.2)

            # Chart 2: Replica comparison (no multiplier)
            ax2.step(
                x,
                hpa_r,
                color=C["red"],
                lw=1.5,
                where="post",
                label="HPA",
            )
            ax2.step(
                x,
                ml_r,
                color=C["blue"],
                lw=2,
                where="post",
                label="ML",
            )
            ax2.set_ylabel("Replicas")
            ax2.legend(loc="upper right", fontsize=7)
            ax2.grid(True, alpha=0.2)
            ax2.set_title("Replica Count Comparison", fontsize=9)

            # Chart 3: Replica diff
            diff = [m - h for m, h in zip(ml_r, hpa_r)]
            colors_bar = [
                C["green"] if d < 0 else C["red"] if d > 0 else C["text_dim"] for d in diff
            ]
            ax3.bar(x, diff, color=colors_bar, alpha=0.7, width=1.0)
            ax3.axhline(0, color=C["text_dim"], lw=0.5)
            ax3.set_ylabel("ML - HPA")
            ax3.set_xlabel("Step")
            ax3.grid(True, alpha=0.2)
            ax3.set_title(
                "Replica Difference (negative = ML uses fewer)",
                fontsize=9,
            )

        self._exp_fig.tight_layout()
        self._exp_canvas.draw_idle()

        # KPI cards
        kpi_data = [
            (
                f"{metrics.hpa_avg_replicas:.1f}",
                f"{metrics.ml_avg_replicas:.1f}",
            ),
            (
                f"{metrics.hpa_replica_minutes:.0f}",
                f"{metrics.ml_replica_minutes:.0f}",
            ),
            (
                f"{metrics.hpa_sla_violations}",
                f"{metrics.ml_sla_violations}",
            ),
            (
                f"{metrics.hpa_scaling_events}",
                f"{metrics.ml_scaling_events}",
            ),
            (
                f"{metrics.hpa_over_provisioning_pct:.1f}%",
                f"{metrics.ml_over_provisioning_pct:.1f}%",
            ),
            (
                f"{metrics.hpa_under_provisioning_pct:.1f}%",
                f"{metrics.ml_under_provisioning_pct:.1f}%",
            ),
            (
                f"{metrics.hpa_avg_reaction_time:.1f}",
                f"{metrics.ml_avg_reaction_time:.1f}",
            ),
            (
                "\u2014",
                f"{metrics.cost_savings_pct:+.1f}%",
            ),
        ]
        for i, (hpa_val, ml_val) in enumerate(kpi_data):
            self._kpi_cards[i].set_values(hpa_val, ml_val, better=self._kpi_better[i])

    def _update_log_tab(self) -> None:
        """Update the activity log / notification feed tab."""
        # Clear and rebuild (only last 30 for performance)
        for widget in self._notif_frame.winfo_children():
            widget.destroy()
        for text, level in self._notifications[-30:]:
            NotificationItem(self._notif_frame, text, level).pack(fill="x", padx=4, pady=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the experiment dashboard application."""
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
