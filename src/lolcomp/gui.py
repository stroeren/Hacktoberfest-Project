"""Desktop app: pick five champions and see the evaluation.

    python -m lolcomp.gui                          # uses config.yaml
    python -m lolcomp.gui --config config.cpu.yaml

"Rules" runs the rule-based labeler (instant, no GPU). "Model" and "Compare" load
the fine-tuned Gemma adapter on first use, in the background.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
import tkinter as tk
from tkinter import ttk

from .analysis import DEFAULT_THRESHOLDS, ISSUE_TITLES, evaluate_team
from .champions import RATED_FIELDS, ROLES, ChampionDB, normalize_name
from .config import load_config
from .prompts import parse_output, user_message

BG = "#0E1621"
HEADER = "#111B27"
PANEL = "#16212E"
BORDER = "#2A3A4C"
SELECTED = "#24303D"
TEXT = "#E8E6E1"
MUTED = "#94A3B3"
GOLD = "#C8AA6E"
GOLD_HOVER = "#E3CB93"
LOW = "#E58A4F"
OK = "#9AA8B6"
HIGH = "#5AA9E6"
VERDICT_COLORS = {"good": "#3FB68B", "average": "#E0A84A", "bad": "#E5604F"}
RATING_COLORS = {"low": LOW, "ok": OK, "high": HIGH}
MIX_COLORS = {"AD-heavy": LOW, "AP-heavy": HIGH, "balanced": OK}

FONT = "Segoe UI"
ROLE_LABELS = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "adc": "ADC", "support": "Support"}
STAT_LABELS = {"damage": "Damage", "cc": "CC", "frontline": "Frontline", "engage": "Engage", "peel": "Peel", "waveclear": "Waveclear"}


def heading(parent: tk.Widget, text: str) -> tk.Label:
    return tk.Label(parent, text=text.upper(), bg=PANEL, fg=GOLD, font=(FONT, 10, "bold"), anchor="w")


def card(parent: tk.Widget, **pack) -> tk.Frame:
    frame = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=14)
    frame.pack(fill="x", **pack)
    return frame


def button(parent: tk.Widget, text: str, command, primary: bool = False) -> tk.Button:
    bg, fg, hover = (GOLD, BG, GOLD_HOVER) if primary else (PANEL, TEXT, SELECTED)
    btn = tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
        relief="flat", bd=0, cursor="hand2", font=(FONT, 12 if primary else 10, "bold" if primary else "normal"),
        highlightbackground=BORDER, highlightthickness=0 if primary else 1, pady=10 if primary else 7,
    )
    btn.bind("<Enter>", lambda _: btn.configure(bg=hover))
    btn.bind("<Leave>", lambda _: btn.configure(bg=bg))
    return btn


def team_stats(db: ChampionDB, team: dict[str, str]) -> tuple[dict[str, int], float]:
    """Team totals per rated field and the AD share, as analysis.evaluate_team computes them."""
    champs = [db.get(team[r]) for r in ROLES]
    totals = {stat: sum(getattr(c, stat) for c in champs) for stat in RATED_FIELDS}
    ad = sum(c.damage_split()[0] for c in champs)
    ap = sum(c.damage_split()[1] for c in champs)
    return totals, ad / (ad + ap) if ad + ap else 0.5


class ResultView(tk.Frame):
    """Scrollable panel that renders one evaluation dict."""

    def __init__(self, parent: tk.Widget, thresholds: dict):
        super().__init__(parent, bg=BG)
        self.thresholds = thresholds
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=BG)
        window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width - 4))
        self.canvas.bind("<Enter>", lambda _: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda _: self.canvas.unbind_all("<MouseWheel>"))
        self.message("Pick a team and press Evaluate.")

    def _on_wheel(self, event: tk.Event) -> None:
        if self.body.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _clear(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        self.canvas.yview_moveto(0)

    def message(self, text: str, color: str = MUTED) -> None:
        self._clear()
        box = card(self.body)
        tk.Label(box, text=text, bg=PANEL, fg=color, font=(FONT, 11), justify="left", anchor="w",
                 wraplength=760).pack(fill="x")

    def show(self, result: dict, totals: dict[str, int] | None = None, ad_share: float | None = None) -> None:
        """totals/ad_share are only known for rule-based results; model results show ratings alone."""
        self._clear()
        ratings = result.get("ratings", {})
        verdict = str(result.get("verdict", "?")).lower()
        score = result.get("score", 0) if isinstance(result.get("score"), (int, float)) else 0

        top = card(self.body)
        tk.Label(top, text=f" {verdict.upper()} ", bg=VERDICT_COLORS.get(verdict, OK), fg=BG,
                 font=(FONT, 18, "bold"), padx=12, pady=4).pack(side="left")
        score_box = tk.Frame(top, bg=PANEL, padx=20)
        score_box.pack(side="left")
        tk.Label(score_box, text=f"Score  {score} / 100", bg=PANEL, fg=TEXT, font=(FONT, 11)).pack(anchor="w")
        bar = tk.Canvas(score_box, width=180, height=8, bg=BG, highlightthickness=0)
        bar.pack(anchor="w", pady=(6, 0))
        bar.create_rectangle(0, 0, max(2, 180 * score / 100), 8, fill=VERDICT_COLORS.get(verdict, OK), width=0)
        tk.Label(top, text=result.get("summary", ""), bg=PANEL, fg=TEXT, font=(FONT, 11), justify="left",
                 anchor="w", wraplength=420).pack(side="left", fill="x", expand=True)

        row = tk.Frame(self.body, bg=BG)
        row.pack(fill="x", pady=12)
        row.columnconfigure((0, 1), weight=1, uniform="col")
        self._totals_card(row, ratings, totals).grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._mix_card(row, ratings, result.get("strengths", []), ad_share).grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        issues = result.get("issues", [])
        box = card(self.body)
        heading(box, f"Issues · {len(issues)}").pack(fill="x", pady=(0, 6))
        if not issues:
            tk.Label(box, text="No major weaknesses.", bg=PANEL, fg=MUTED, font=(FONT, 10), anchor="w").pack(fill="x")
        for issue in issues:
            line = tk.Frame(box, bg=PANEL)
            line.pack(fill="x", pady=4)
            title = ISSUE_TITLES.get(issue.get("code", ""), issue.get("code", "?"))
            tk.Label(line, text=title, bg=PANEL, fg=VERDICT_COLORS["bad"], font=(FONT, 10, "bold"), width=22,
                     anchor="nw", justify="left", wraplength=170).pack(side="left", anchor="n")
            tk.Label(line, text=issue.get("detail", ""), bg=PANEL, fg=TEXT, font=(FONT, 10), anchor="w",
                     justify="left", wraplength=560).pack(side="left", fill="x", expand=True)

    def _totals_card(self, parent: tk.Widget, ratings: dict, totals: dict | None) -> tk.Frame:
        box = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=14)
        heading(box, "Team totals · 0–15" if totals else "Ratings").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        width, height = 190, 10
        for i, stat in enumerate(RATED_FIELDS, start=1):
            rating = str(ratings.get(stat, "?")).lower()
            color = RATING_COLORS.get(rating, OK)
            tk.Label(box, text=STAT_LABELS[stat], bg=PANEL, fg=MUTED, font=(FONT, 10), anchor="w", width=10).grid(row=i, column=0, sticky="w", pady=3)
            bar = tk.Canvas(box, width=width, height=height + 6, bg=PANEL, highlightthickness=0)
            bar.grid(row=i, column=1, padx=6)
            bar.create_rectangle(0, 3, width, height + 3, fill=BG, width=0)
            if totals:
                value = totals[stat]
                bar.create_rectangle(0, 3, max(2, width * value / 15), height + 3, fill=color, width=0)
                for cut in self.thresholds[stat]:
                    x = width * cut / 15
                    bar.create_line(x, 0, x, height + 6, fill=MUTED, width=2)
            else:
                value = ""
                bar.create_rectangle(0, 3, width * {"low": 0.2, "ok": 0.5, "high": 0.85}.get(rating, 0), height + 3, fill=color, width=0)
            tk.Label(box, text=value, bg=PANEL, fg=TEXT, font=(FONT, 10), width=3, anchor="e").grid(row=i, column=2)
            tk.Label(box, text=rating.upper(), bg=PANEL, fg=color, font=(FONT, 9, "bold"), width=5, anchor="w").grid(row=i, column=3, padx=(6, 0))
        if totals:
            tk.Label(box, text="Ticks mark the low / high cutoffs.", bg=PANEL, fg=MUTED, font=(FONT, 8)).grid(row=8, column=0, columnspan=4, sticky="w", pady=(8, 0))
        return box

    def _mix_card(self, parent: tk.Widget, ratings: dict, strengths: list, ad_share: float | None) -> tk.Frame:
        box = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=14)
        heading(box, "Damage mix").pack(fill="x")
        mix = ratings.get("damage_mix", "?")
        if ad_share is not None:
            width = 300
            bar = tk.Canvas(box, width=width, height=22, bg=PANEL, highlightthickness=0)
            bar.pack(anchor="w", pady=(8, 4))
            split = width * ad_share
            bar.create_rectangle(0, 0, split, 22, fill=LOW, width=0)
            bar.create_rectangle(split, 0, width, 22, fill=HIGH, width=0)
            if ad_share >= 0.15:
                bar.create_text(8, 11, text=f"AD {round(ad_share * 100)}%", anchor="w", fill=BG, font=(FONT, 9, "bold"))
            if ad_share <= 0.85:
                bar.create_text(width - 8, 11, text=f"AP {round((1 - ad_share) * 100)}%", anchor="e", fill=BG, font=(FONT, 9, "bold"))
        tk.Label(box, text=mix, bg=PANEL, fg=MIX_COLORS.get(mix, OK), font=(FONT, 10, "bold"), anchor="w").pack(fill="x", pady=(4, 10))

        heading(box, "Power curve").pack(fill="x")
        curve = tk.Frame(box, bg=PANEL)
        curve.pack(fill="x", pady=(8, 10))
        for i, phase in enumerate(("early", "balanced", "late")):
            active = ratings.get("power_curve") == phase
            tk.Label(curve, text=phase.capitalize(), bg=SELECTED if active else PANEL, fg=TEXT if active else MUTED,
                     font=(FONT, 10, "bold" if active else "normal"), pady=5,
                     highlightbackground=GOLD if active else BORDER, highlightthickness=1).grid(row=0, column=i, sticky="ew", padx=2)
            curve.columnconfigure(i, weight=1, uniform="phase")

        heading(box, "Strengths").pack(fill="x")
        if not strengths:
            tk.Label(box, text="None found.", bg=PANEL, fg=MUTED, font=(FONT, 10), anchor="w").pack(fill="x", pady=(6, 0))
        for s in strengths:
            tk.Label(box, text=f"•  {s}", bg=PANEL, fg=TEXT, font=(FONT, 10), anchor="w", justify="left",
                     wraplength=320).pack(fill="x", pady=(6, 0))
        return box


class App(tk.Tk):
    def __init__(self, cfg: dict, adapter: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter or cfg["train"]["output_dir"]
        self.db = ChampionDB.load(cfg["data"]["champions"])
        self.thresholds = {**DEFAULT_THRESHOLDS, **self.db.thresholds}
        self.model = None  # (model, tokenizer) once loaded
        self.busy = False

        self.title("LoL Comp Evaluator")
        self.configure(bg=BG)
        self.geometry("1200x860")
        self.minsize(1080, 640)
        self._style()

        self._header()
        self.status = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status, bg=HEADER, fg=MUTED, font=(FONT, 9), anchor="w", padx=24,
                 pady=6).pack(side="bottom", fill="x")
        content = tk.Frame(self, bg=BG, padx=24, pady=18)
        content.pack(fill="both", expand=True)
        self._sidebar(content)
        self._results(content)
        self.bind("<Return>", lambda _: self.evaluate())

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=BG, background=PANEL, foreground=TEXT, arrowcolor=MUTED,
                        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=6, insertcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", BG)], bordercolor=[("focus", GOLD)],
                  lightcolor=[("focus", GOLD)])
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", GOLD)
        self.option_add("*TCombobox*Listbox.selectForeground", BG)
        self.option_add("*TCombobox*Listbox.font", (FONT, 10))
        style.configure("TNotebook", background=BG, borderwidth=1, tabmargins=0, bordercolor=BORDER,
                        lightcolor=BG, darkcolor=BG)
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(16, 7), borderwidth=1,
                        font=(FONT, 10), lightcolor=BG, darkcolor=BG, bordercolor=BORDER, focuscolor=PANEL)
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEXT)],
                  lightcolor=[("selected", GOLD)], expand=[("selected", (0, 0, 0, 0))])
        style.configure("Vertical.TScrollbar", background=PANEL, troughcolor=BG, bordercolor=BG, arrowcolor=MUTED,
                        lightcolor=PANEL, darkcolor=PANEL)

    def _header(self) -> None:
        bar = tk.Frame(self, bg=HEADER, padx=24, pady=14)
        bar.pack(fill="x")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        logo = tk.Canvas(bar, width=30, height=30, bg=HEADER, highlightthickness=0)
        logo.create_polygon(15, 2, 27, 8.5, 27, 21.5, 15, 28, 3, 21.5, 3, 8.5, outline=GOLD, fill="", width=2)
        logo.create_line(15, 9, 15, 22, fill=GOLD, width=2)
        logo.pack(side="left", padx=(0, 12))
        titles = tk.Frame(bar, bg=HEADER)
        titles.pack(side="left")
        tk.Label(titles, text="LoL Comp Evaluator", bg=HEADER, fg=TEXT, font=(FONT, 15, "bold")).pack(anchor="w")
        tk.Label(titles, text="Rule-based labeler · Gemma 4 QLoRA adapter", bg=HEADER, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        tk.Label(bar, text=f"{len(self.db.champions)} champions loaded · {self.cfg['data']['champions']}",
                 bg=HEADER, fg=MUTED, font=(FONT, 9)).pack(side="right")

    def _sidebar(self, parent: tk.Widget) -> None:
        side = tk.Frame(parent, bg=BG, width=310)
        side.pack(side="left", fill="y", padx=(0, 20))
        side.pack_propagate(False)

        team_box = card(side)
        heading(team_box, "Team").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.picks: dict[str, tk.StringVar] = {}
        self.boxes: dict[str, ttk.Combobox] = {}
        for i, role in enumerate(ROLES, start=1):
            tk.Label(team_box, text=ROLE_LABELS[role], bg=PANEL, fg=MUTED, font=(FONT, 10), width=8, anchor="w").grid(row=i, column=0, sticky="w", pady=5)
            var = tk.StringVar()
            box = ttk.Combobox(team_box, textvariable=var, values=self._choices(role), font=(FONT, 11), width=18)
            box.grid(row=i, column=1, sticky="ew", pady=5)
            box.bind("<KeyRelease>", lambda e, r=role: self._filter(r, e))
            self.picks[role], self.boxes[role] = var, box
        team_box.columnconfigure(1, weight=1)

        mode_box = card(side, pady=16)
        heading(mode_box, "Evaluate with").pack(fill="x", pady=(0, 8))
        self.mode = tk.StringVar(value="rules")
        seg = tk.Frame(mode_box, bg=BORDER, padx=1, pady=1)
        seg.pack(fill="x")
        self.mode_buttons: dict[str, tk.Radiobutton] = {}
        for i, (value, label) in enumerate((("rules", "Rules"), ("model", "Model"), ("compare", "Compare"))):
            self.mode_buttons[value] = tk.Radiobutton(seg, text=label, value=value, variable=self.mode, indicatoron=False, bg=BG, fg=TEXT,
                           selectcolor=GOLD, activebackground=SELECTED, activeforeground=TEXT, font=(FONT, 10),
                           relief="flat", bd=0, pady=7, cursor="hand2", command=self._mode_changed,
                           )
            self.mode_buttons[value].grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 1, 0))
            seg.columnconfigure(i, weight=1, uniform="seg")
        self.mode_hint = tk.Label(mode_box, bg=PANEL, fg=MUTED, font=(FONT, 9), justify="left", anchor="w", wraplength=270)
        self.mode_hint.pack(fill="x", pady=(10, 0))
        self._mode_changed()

        self.eval_button = button(side, "EVALUATE", self.evaluate, primary=True)
        self.eval_button.pack(fill="x")
        row = tk.Frame(side, bg=BG)
        row.pack(fill="x", pady=10)
        row.columnconfigure((0, 1), weight=1, uniform="btn")
        button(row, "Random comp", self.randomize).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        button(row, "Clear", self.clear).grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _results(self, parent: tk.Widget) -> None:
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(side="left", fill="both", expand=True)
        self.rules_view = ResultView(self.tabs, self.thresholds)
        self.model_view = ResultView(self.tabs, self.thresholds)
        self.tabs.add(self.rules_view, text="Rule-based")
        self.tabs.add(self.model_view, text="Model")

    def _choices(self, role: str) -> list[str]:
        """Champions played in this role first, then everyone else."""
        main = sorted(c.name for c in self.db.for_role(role))
        rest = sorted(c.name for c in self.db.champions if role not in c.roles)
        return main + rest

    def _filter(self, role: str, event: tk.Event) -> None:
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab"):
            return
        typed = normalize_name(self.picks[role].get())
        choices = self._choices(role)
        self.boxes[role]["values"] = [n for n in choices if typed in normalize_name(n)] or choices

    def _mode_changed(self) -> None:
        hints = {
            "rules": "Instant, no GPU. Every champion must be in champions.json.",
            "model": f"Loads the adapter from {self.adapter} on first use (GPU recommended).",
            "compare": "Runs both and shows them side by side in the tabs.",
        }
        self.mode_hint.configure(text=hints[self.mode.get()])
        for value, btn in self.mode_buttons.items():
            btn.configure(fg=BG if value == self.mode.get() else TEXT,
                          font=(FONT, 10, "bold" if value == self.mode.get() else "normal"))

    def randomize(self) -> None:
        taken: set[str] = set()
        for role in ROLES:
            pool = [c.name for c in self.db.for_role(role) if c.name not in taken]
            pick = random.choice(pool)
            taken.add(pick)
            self.picks[role].set(pick)
            self.boxes[role]["values"] = self._choices(role)

    def clear(self) -> None:
        for role in ROLES:
            self.picks[role].set("")
            self.boxes[role]["values"] = self._choices(role)
        self.rules_view.message("Pick a team and press Evaluate.")
        self.model_view.message("Pick a team and press Evaluate.")
        self.status.set("Ready")

    def _team(self) -> dict[str, str] | None:
        team = {}
        for role in ROLES:
            name = self.picks[role].get().strip()
            if not name:
                self.status.set(f"Pick a champion for {ROLE_LABELS[role]}.")
                self.boxes[role].focus_set()
                return None
            champ = self.db.get(name)
            team[role] = champ.name if champ else name  # unknown names still go to the model
            self.picks[role].set(team[role])
        return team

    def evaluate(self) -> None:
        if self.busy:
            return
        team = self._team()
        if team is None:
            return
        mode = self.mode.get()
        if mode in ("rules", "compare"):
            self._run_rules(team)
        if mode in ("model", "compare"):
            self._run_model(team, select=mode == "model")

    def _run_rules(self, team: dict[str, str]) -> None:
        self.tabs.select(self.rules_view)
        unknown = [name for name in team.values() if self.db.get(name) is None]
        if unknown:
            self.rules_view.message(f"Rules need every champion in champions.json. Not found: {', '.join(unknown)}.\n"
                                    "Use Model mode for champions outside the file.", VERDICT_COLORS["average"])
            self.status.set("Rule-based evaluation skipped")
            return
        start = time.perf_counter()
        result = evaluate_team({r: self.db.get(n) for r, n in team.items()}, self.db.thresholds).to_dict()
        totals, ad_share = team_stats(self.db, team)
        self.rules_view.show(result, totals, ad_share)
        self.status.set(f"Rule-based result in {(time.perf_counter() - start) * 1000:.0f} ms")

    def _run_model(self, team: dict[str, str], select: bool) -> None:
        if select:
            self.tabs.select(self.model_view)
        self.busy = True
        self.eval_button.configure(state="disabled", text="WORKING…")
        self.model_view.message("Loading model…" if self.model is None else "Generating…")
        self.status.set("Loading model (this can take a minute)…" if self.model is None else "Generating…")
        threading.Thread(target=self._model_worker, args=(team,), daemon=True).start()

    def _model_worker(self, team: dict[str, str]) -> None:
        try:
            from .modeling import generate, load_model  # heavy imports only when needed

            if self.model is None:
                self.model = load_model(self.cfg["model"], adapter=self.adapter)
                self.after(0, self.status.set, "Generating…")
            model, tokenizer = self.model
            start = time.perf_counter()
            [reply] = generate(model, tokenizer, [[{"role": "user", "content": user_message(team)}]])
            elapsed = time.perf_counter() - start
            self.after(0, self._model_done, reply, elapsed, None)
        except Exception as exc:  # surface any load/generation failure in the UI
            self.after(0, self._model_done, None, 0.0, exc)

    def _model_done(self, reply: str | None, elapsed: float, error: Exception | None) -> None:
        self.busy = False
        self.eval_button.configure(state="normal", text="EVALUATE")
        if error is not None:
            self.model_view.message(f"Model failed: {error}", VERDICT_COLORS["bad"])
            self.status.set("Model failed")
            return
        parsed = parse_output(reply)
        if parsed is None:
            self.model_view.message(f"Could not parse the model's JSON. Raw reply:\n\n{reply}", VERDICT_COLORS["average"])
        else:
            self.model_view.show(parsed)
        self.status.set(f"Model result in {elapsed:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", help="LoRA adapter dir (default: train.output_dir from config)")
    args = parser.parse_args()
    App(load_config(args.config), adapter=args.adapter).mainloop()


if __name__ == "__main__":
    main()
