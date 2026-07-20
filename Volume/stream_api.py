"""
Потоковый интерфейс к volume_core: функция add_pts(pts, move) для работы
в реальном времени на установке.

Использование:

    from stream_api import VolumeStream

    vs = VolumeStream([1, 0, 0], belt_level=-3.9,
                      target_volume=50.0)      # None = резка отключена

    # на каждом кадре сканера:
    event = vs.add_pts(pts, move)
    if event is None:
        pass                        # обычный кадр, копим
    elif isinstance(event, CutParams):
        ...                         # достигнут целевой объём: резать!
    elif isinstance(event, DetailParams):
        ...                         # деталь полностью проехала камеры

    # в конце потока:
    for event in vs.finish():
        ...

Событий за один вызов может быть несколько (например, рез и сразу конец
детали) — тогда возвращается список.
"""

from collections import namedtuple
from volume_core import ConveyorVolume

# деталь полностью проехала зону камер: итоговый объём и номер
DetailParams = namedtuple("DetailParams", ["V", "id"])

# накопленный объём достиг target_volume:
#   V              -- объём порции (равен target_volume)
#   id             -- номер детали
#   len_from_start -- расстояние от переднего края детали до точки реза
#   len_from_frame -- расстояние от точки реза до последнего кадра лазера
#                     (насколько сканер уже проехал мимо точки реза)
CutParams = namedtuple("CutParams", ["V", "id",
                                     "len_from_start", "len_from_frame"])


class VolumeStream:
    """Обёртка над ConveyorVolume: вместо печати и колбэков возвращает
    события из add_pts, как требует потоковый интерфейс установки."""

    def __init__(self, direction, target_volume=None, **kwargs):
        """direction и target_volume обязательны по смыслу, остальные
        параметры (belt_level, slice_step, min_gap, profile_mode и т.д.)
        передаются в ConveyorVolume как есть."""
        self._pending = []
        self._core = ConveyorVolume(
            direction,
            target_volume=target_volume,
            on_object=self._on_detail,
            on_cut=self._on_cut,
            **kwargs)

    # ------------------- публичный API -------------------

    def add_pts(self, pts, move):
        """Обработать кадр. pts -- (N, 3) точки, move -- смещение ленты.
        Возвращает None (обычный кадр), DetailParams, CutParams
        или список событий, если их случилось несколько."""
        self._core.add_frame(pts, move)
        return self._pop_events()

    def finish(self):
        """Завершить поток: принудительно закрыть активные детали.
        Возвращает список оставшихся событий."""
        self._core.finalize()
        events = self._pop_events()
        if events is None:
            return []
        return events if isinstance(events, list) else [events]

    @property
    def volumes(self):
        """Готовые объёмы: словарь id -> V."""
        return dict(self._core.volumes)

    # ------------------- внутреннее -------------------

    def _on_detail(self, fig_id, volume):
        self._pending.append(DetailParams(V=volume, id=fig_id))

    def _on_cut(self, fig_id, volume, len_start, len_frame):
        self._pending.append(CutParams(V=volume, id=fig_id,
                                       len_from_start=len_start,
                                       len_from_frame=len_frame))

    def _pop_events(self):
        if not self._pending:
            return None
        events, self._pending = self._pending, []
        return events[0] if len(events) == 1 else events