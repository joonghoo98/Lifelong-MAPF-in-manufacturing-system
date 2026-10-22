"""The live viewer: a pygame-ce window that advances a Simulation and draws it, frame by frame (user decision V1).

Each frame handles the keys, advances the simulation by the wall time since the last frame times the speed (at
most 60 s of simulated time per frame; the first frame only starts the clock), and draws the scene of that time.
The events are run by the simulation's own kernel, so the drawing speed never changes the results. The map is
drawn once onto a background surface.

Keys: S speed, W warm-up, T sim time (type a number, Enter; Esc cancels), F fast run (F again stops it), Space
pause, + / - speed step, . one dt step while paused, Esc quit. A fast run computes event to event without pacing, up
to warm-up + sim time, then goes on at the set speed (user decision, 2026-10-22); it draws and reads the keys about
every 0.1 s of wall time, which only decides when to draw, never what happens. The KPI lines are those of
[warm-up end, now], as DAIM measures after its warm-up.
"""
import math
import time
from collections.abc import Callable

import pygame

from agvtb.viz.scene import Scene, build_scene

SPEEDS = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
MAX_STEP_S = 60.0
FAST_STEP_S = 10.0  # a fast run computes in steps of this much simulated time ...
FAST_FRAME_S = 0.1  # ... and draws a frame after about this much wall time
FPS = 60
KIND_COLOR = {"move": (40, 170, 60), "rotate": (0, 160, 170), "blocked": (210, 40, 40), "dwell": (240, 150, 20),
              "rest": (130, 130, 130)}
BACK, EDGE, NODE, PORT, SIDING, TEXT = (250, 250, 250), (205, 205, 205), (170, 170, 170), (30, 70, 200), \
    (130, 90, 170), (20, 20, 20)
CYCLE_COLOR = (230, 0, 230)  # vehicles in a wait cycle: a colour no state uses
MIN_BODY_PX = 8  # a vehicle is drawn at least this big, so its colour shows on big maps
KEYS_UP = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)  # '+' on a Korean or US keyboard is K_EQUALS
KEYS_DOWN = (pygame.K_MINUS, pygame.K_KP_MINUS)
KEYS_STEP = (pygame.K_PERIOD, pygame.K_KP_PERIOD)
KEYS_ENTER = (pygame.K_RETURN, pygame.K_KP_ENTER)
DIGITS = {getattr(pygame, f"K_{d}"): str(d) for d in range(10)} | {getattr(pygame, f"K_KP{d}"): str(d) for d in range(10)}
FIELDS = {pygame.K_s: "S", pygame.K_w: "W", pygame.K_t: "T"}
FIELD_NAME = {"S": "speed", "W": "warm-up s", "T": "sim time s"}


class Viewer:
    def __init__(self, sim, size: tuple[int, int] = (1280, 800), speed: float = 10.0, warmup: float = 3600.0,
                 sim_time: float = 86400.0):
        self.sim = sim
        self.size = size
        self.speed = speed
        self.warmup = warmup
        self.sim_time = sim_time
        self.paused = False
        self.editing: str | None = None  # "S", "W" or "T" while a number is typed
        self.buffer = ""
        self.fast_target: float | None = None
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(f"agvtb live viewer - {sim.world.name}")
        self.font = pygame.font.SysFont(None, 16)
        self._ticker = pygame.time.Clock()
        self._fit()
        self.background = self._draw_background()

    # --- geometry ----------------------------------------------------------------------------------------------
    def _fit(self) -> None:
        xs = [x for x, _ in self.sim.world.nodes.values()]
        ys = [y for _, y in self.sim.world.nodes.values()]
        self._x0, self._y1 = min(xs), max(ys)
        w, h = self.size
        span_x, span_y = max(max(xs) - self._x0, 1e-9), max(self._y1 - min(ys), 1e-9)
        self.scale = min(w * 0.9 / span_x, h * 0.9 / span_y)
        self._ox = (w - span_x * self.scale) / 2
        self._oy = (h - span_y * self.scale) / 2

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self._ox + (x - self._x0) * self.scale, self._oy + (self._y1 - y) * self.scale

    def _draw_background(self) -> pygame.Surface:
        world = self.sim.world
        surface = pygame.Surface(self.size)
        surface.fill(BACK)
        drawn = set()
        for (u, v) in sorted(world.segments):
            if (v, u) in drawn:
                continue
            drawn.add((u, v))
            pygame.draw.line(surface, EDGE, self.to_screen(*world.nodes[u]), self.to_screen(*world.nodes[v]), 1)
        for x, y in world.nodes.values():
            pygame.draw.circle(surface, NODE, self.to_screen(x, y), 1)
        r = max(3, int(0.3 * self.scale))
        for s in self.sim.sidings:
            pygame.draw.circle(surface, SIDING, self.to_screen(*world.nodes[s]), r, 1)
        for p in sorted(set(world.port_nodes)):
            sx, sy = self.to_screen(*world.nodes[p])
            pygame.draw.rect(surface, PORT, pygame.Rect(sx - r, sy - r, 2 * r, 2 * r), 2)
        return surface

    # --- drawing -----------------------------------------------------------------------------------------------
    def _vehicle_polygon(self, x: float, y: float, heading: float) -> list[tuple[float, float]]:
        fp = self.sim.world.footprint
        half_l = max(fp.length_m * self.scale, MIN_BODY_PX) / 2
        half_w = max(fp.width_m * self.scale, MIN_BODY_PX) / 2
        cx, cy = self.to_screen(x, y)
        a = math.radians(heading)
        ux, uy = math.cos(a), -math.sin(a)  # screen y grows downwards
        vx, vy = -uy, ux
        return [(cx + sx * half_l * ux + sy * half_w * vx, cy + sx * half_l * uy + sy * half_w * vy)
                for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]

    def _draw(self, scene: Scene, extra: str | None = None) -> None:
        self.screen.blit(self.background, (0, 0))
        for port in scene.ports:
            if port.waiting:
                sx, sy = self.to_screen(port.x, port.y)
                self.screen.blit(self.font.render(str(port.waiting), True, PORT), (sx + 6, sy - 14))
        for v in scene.vehicles:
            poly = self._vehicle_polygon(v.x, v.y, v.heading_deg)
            pygame.draw.polygon(self.screen, KIND_COLOR[v.kind], poly)
            pygame.draw.polygon(self.screen, CYCLE_COLOR if v.in_cycle else TEXT, poly, 3 if v.in_cycle else 1)
            front = ((poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2)
            pygame.draw.line(self.screen, TEXT, self.to_screen(v.x, v.y), front, 2)
            self.screen.blit(self.font.render(f"{v.vid}{v.phase}", True, TEXT), (poly[0][0] + 2, poly[0][1] - 12))
        lines = list(scene.hud) + [self._settings_line()] + ([extra] if extra else [])
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, TEXT), (8, 6 + 16 * i))

    def _settings_line(self) -> str:
        if self.editing:
            return f"{FIELD_NAME[self.editing]} = {self.buffer}_   (Enter to set, Esc to cancel)"
        fast = f"  FAST to t {self.fast_target:.0f} s" if self.fast_target is not None else ""
        return (f"[S] speed x{self.speed:g}  [W] warm-up {self.warmup:g} s  [T] sim time {self.sim_time:g} s  "
                f"[F] fast run{fast}")

    def draw(self, extra: str | None = None) -> None:
        """Draw the scene of now and show it."""
        self._draw(build_scene(self.sim, self.speed, self.paused, warmup=self.warmup), extra)
        pygame.display.flip()

    def save(self, path) -> None:
        pygame.image.save(self.screen, str(path))

    # --- the loop ----------------------------------------------------------------------------------------------
    def _edit(self, key) -> None:
        """A key while a number is typed: digits, '.', Backspace, Enter (set if valid), Esc (cancel)."""
        if key in DIGITS:
            self.buffer += DIGITS[key]
        elif key in KEYS_STEP:
            self.buffer += "."
        elif key == pygame.K_BACKSPACE:
            self.buffer = self.buffer[:-1]
        elif key in KEYS_ENTER:
            try:
                value = float(self.buffer)
            except ValueError:
                value = math.nan
            ok = math.isfinite(value) and (value >= 0 if self.editing == "W" else value > 0)
            if ok:
                if self.editing == "S":
                    self.speed = value
                elif self.editing == "W":
                    self.warmup = value
                else:
                    self.sim_time = value
            self.editing, self.buffer = None, ""
        elif key == pygame.K_ESCAPE:
            self.editing, self.buffer = None, ""

    def _handle(self, event) -> str | None:
        if event.type == pygame.QUIT:
            return "quit"
        if event.type != pygame.KEYDOWN:
            return None
        if self.editing:
            self._edit(event.key)
            return None
        if event.key in FIELDS:
            self.editing, self.buffer = FIELDS[event.key], ""
        elif event.key == pygame.K_f and self.fast_target is not None:
            self.fast_target = None
        elif event.key == pygame.K_f:
            target = self.warmup + self.sim_time
            if target > self.sim.now:
                self.fast_target, self.paused = target, False
        elif event.key == pygame.K_ESCAPE:
            return "quit"
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key in KEYS_UP:
            self.speed = next((s for s in SPEEDS if s > self.speed), self.speed)
        elif event.key in KEYS_DOWN:
            self.speed = next((s for s in reversed(SPEEDS) if s < self.speed), self.speed)
        elif event.key in KEYS_STEP and self.paused:
            self.sim.advance_to(self.sim.now + self.sim.cfg.dt)
        return None

    def run(self, max_frames: int | None = None, clock: Callable[[], float] = time.perf_counter) -> str:
        """Advance and draw until the window is closed ("quit"), the run ends ("end"), it freezes ("frozen") or
        max_frames frames were drawn ("frames")."""
        last, frames = None, 0
        while True:
            for event in pygame.event.get():
                reason = self._handle(event)
                if reason:
                    return reason
            now = clock()
            if self.paused:
                pass
            elif self.fast_target is not None:  # event to event, no pacing; a frame now and then for progress
                deadline = now + FAST_FRAME_S
                while True:
                    self.sim.advance_to(min(self.fast_target, self.sim.now + FAST_STEP_S))
                    if self.sim.now >= min(self.fast_target, self.sim.end) or self.sim.stopped:
                        self.fast_target = None
                        break
                    if clock() >= deadline:
                        break
                now = clock()  # the compute time is not paced in the next frame (final review I1)
            elif last is not None:
                self.sim.advance_to(self.sim.now + min(MAX_STEP_S, (now - last) * self.speed))
            last = now
            self.draw()
            frames += 1
            if self.sim.stopped:
                return "frozen"
            if self.sim.now >= self.sim.end:
                return "end"
            if max_frames is not None and frames >= max_frames:
                return "frames"
            self._ticker.tick(FPS)

    def hold(self) -> str:
        """Keep the last frame on screen until Esc or the window is closed (to look at how the run ended)."""
        self.draw("ended - Esc to close")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return "quit"
            self._ticker.tick(30)
