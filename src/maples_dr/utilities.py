from __future__ import annotations

__all__ = ["RichProgress"]

from functools import reduce
from time import time
from typing import Iterable, NamedTuple, Tuple, TypeGuard

from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class RichProgress:
    def __init__(self, name, total, done_message=None, columns=()) -> None:
        self.progress = Progress(*columns, transient=False)
        self.name = name
        self.total = total
        self.done_message = done_message
        self.task = None
        self.t0 = None

    def __enter__(self):
        self.task = self.progress.add_task(self.name, total=self.total)
        self.t0 = time()
        self.progress.start()
        return self

    def update(self, value=1, message=None):
        self.progress.update(self.task, advance=value, description=message)

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time() - self.t0
        if exc_type is None:
            self.progress.update(self.task, visible=False)
            self.progress.remove_task(self.task)
            if self.done_message is not None:
                self.progress.console.print(self.done_message.replace("{t}", f"{elapsed:.1f}"))
        self.progress.stop()

    @staticmethod
    def iteration(name, total, done_message=None):
        return RichProgress(
            name,
            total,
            done_message,
            columns=(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ),
        )

    @staticmethod
    def download(item_name, byte_size):
        return RichProgress(
            f"Downloading {item_name}...",
            byte_size,
            item_name + " downloaded in {t} seconds",
            columns=(
                TextColumn("{task.description}"),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ),
        )


class Rect(NamedTuple):
    h: float
    w: float
    y: float = 0
    x: float = 0

    @property
    def center(self) -> Point:
        return Point(self.y + self.h // 2, self.x + self.w // 2)

    @property
    def top_left(self) -> Point:
        return Point(self.y, self.x)

    @property
    def bottom_right(self) -> Point:
        return Point(self.y + self.h, self.x + self.w)

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def shape(self) -> Point:
        return Point(y=self.h, x=self.w)

    @property
    def area(self) -> float:
        return self.h * self.w

    def to_int(self):
        return Rect(*(int(round(_)) for _ in (self.h, self.w, self.y, self.x)))

    @classmethod
    def from_size(cls, shape: Tuple[float, float]):
        return cls(shape[0], shape[1])

    @classmethod
    def from_points(cls, p1: Tuple[float, float], p2: Tuple[float, float], ensure_positive: bool = False):
        if not ensure_positive:
            return cls(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]), min(p1[0], p2[0]), min(p1[1], p2[1]))
        else:
            rect = cls(p2[0] - p1[0], p2[1] - p1[1], p1[0], p1[1])
            return Rect.empty() if rect.h < 0 or rect.w < 0 else rect

    @classmethod
    def from_center(cls, center: Tuple[float, float], shape: Tuple[float, float]):
        return cls(shape[0], shape[1], center[0] - shape[0] // 2, center[1] - shape[1] // 2)

    @classmethod
    def empty(cls):
        return cls(0, 0, 0, 0)

    def is_self_empty(self) -> bool:
        return self.w == 0 or self.h == 0

    @classmethod
    def is_empty(cls, rect: Rect | None) -> bool:
        if rect is None:
            return True
        if isinstance(rect, tuple) and len(rect) == 4:
            rect = Rect(*rect)
        return isinstance(rect, tuple) and (rect.w == 0 or rect.h == 0)

    @classmethod
    def is_rect(cls, r) -> TypeGuard[Rect]:
        return isinstance(r, Rect) or (isinstance(r, tuple) and len(r) == 4)

    def __repr__(self):
        return "Rect(y={}, x={}, h={}, w={})".format(self.y, self.x, self.h, self.w)

    def __or__(self, other):
        if isinstance(other, Rect):
            if self.is_self_empty():
                return other
            if other.is_self_empty():
                return self
            return Rect.from_points(
                (min(self.top, other.top), min(self.left, other.left)),
                (max(self.bottom, other.bottom), max(self.right, other.right)),
            )
        else:
            raise TypeError("Rect can only be combined only with another Rect")

    def __and__(self, other):
        if isinstance(other, Rect):
            return Rect.from_points(
                (max(self.top, other.top), max(self.left, other.left)),
                (min(self.bottom, other.bottom), min(self.right, other.right)),
                ensure_positive=True,
            )
        else:
            raise TypeError("Rect can only be combined only with another Rect")

    def __bool__(self):
        return not self.is_self_empty()

    def __add__(self, other: Point | float):
        if isinstance(other, float):
            other = Point(other, other)
        if isinstance(other, Point):
            return self.translate(other.y, other.x)
        raise TypeError("Rect can only be translated by a Point or a float")

    def __sub__(self, other: Point | float):
        if isinstance(other, float):
            other = Point(other, other)
        if isinstance(other, Point):
            return self.translate(-other.y, -other.x)
        raise TypeError("Rect can only be translated by a Point or a float")

    def __mul__(self, other: float):
        return self.scale(other)

    def __truediv__(self, other: float):
        return self.scale(1 / other)

    def __contains__(self, other: Point | Rect):
        if isinstance(other, Point):
            return self.y <= other.y <= self.y + self.h and self.x <= other.x <= self.x + self.w
        elif isinstance(other, Rect):
            return not Rect.is_empty(self & other)
        else:
            raise TypeError("Rect can only be compared with a Point or a Rect")

    def translate(self, y: float, x: float):
        return Rect(self.h, self.w, self.y + y, self.x + x)

    def scale(self, fy: float, fx: float | None = None):
        if fx is None:
            fx = fy
        return Rect(self.h * fy, self.w * fx, self.y * fy, self.x * fx)

    def slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)

    def box(self) -> tuple[float, float, float, float]:
        """
        Return the coordinates of the rectangle in the form (left, top, right, bottom).
        """
        return self.x, self.y, self.x + self.w, self.y + self.h

    @staticmethod
    def union(*rects: Tuple[Iterable[Rect] | Rect, ...]) -> Rect:
        rects = sum(((r,) if isinstance(r, Rect) else tuple(r) for r in rects), ())
        return reduce(lambda a, b: a | b, rects)

    @staticmethod
    def intersection(*rects: Tuple[Iterable[Rect] | Rect, ...]) -> Rect:
        rects = sum(((r,) if isinstance(r, Rect) else tuple(r) for r in rects), ())
        return reduce(lambda a, b: a & b, rects)


class Point(NamedTuple):
    y: float
    x: float

    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    def to_int(self) -> Point:
        return Point(int(round(self.y)), int(round(self.x)))

    def __add__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y + other, self.x + other)
        y, x = other
        return Point(self.y + y, self.x + x)

    def __radd__(self, other: Tuple[float, float] | float):
        return self + other

    def __sub__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y - other, self.x - other)
        y, x = other
        return Point(self.y - y, self.x - x)

    def __rsub__(self, other: Tuple[float, float] | float):
        return -(self - other)

    def __mul__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y * other, self.x * other)
        y, x = other
        return Point(self.y * y, self.x * x)

    def __rmul__(self, other: Tuple[float, float] | float):
        return self * other

    def __truediv__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y / other, self.x / other)
        y, x = other
        return Point(self.y / y, self.x / x)

    def __rtruediv__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(other / self.y, other / self.x)
        y, x = other
        return Point(y / self.y, x / self.x)

    def __neg__(self):
        return Point(-self.y, -self.x)
