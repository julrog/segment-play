from __future__ import annotations

from multiprocessing.managers import BaseManager
from typing import Dict, Optional, Tuple, Union


class GameSettings:
    def __init__(self) -> None:
        self.properties: Dict[str, Union[bool, int]] = {
            'save_imgs': False,
            'overall_mirror': True,
            'random_people_mirror': False,
            'form_invisibility': False,
            'all_invisibility': False,
            'hide_background': False,
            'gray_game': False,
            'black': False,
            'show_boxes': False,
            'show_poses': False,
            'segmentation_parts': 1,
            'segmentation_change': False
        }
        self.id_position_map: Dict[int, float] = {}

    def set(self, key: str, value: Union[bool, int]) -> None:
        self.properties.update({key: value})

    def get(self, key: str) -> Union[bool, int]:
        value = self.properties.get(key)
        assert value is not None
        return value

    def has_position_map(self, id: int) -> float:
        return id in self.id_position_map

    def get_position_map(self, id: int) -> float:
        return self.id_position_map.get(id, 0.0)

    def update_position_map(self, id: int, position: float) -> None:
        position_map = self.id_position_map
        position_map.update({id: position})
        self.id_position_map = position_map

    def handle_key(self, key: str) -> None:
        if key == 'm':
            self.set('overall_mirror', not self.get('overall_mirror'))
        if key == 'r':
            self.set('random_people_mirror',
                     not self.get('random_people_mirror'))
        if key == 'f':
            self.set('form_invisibility', not self.get('form_invisibility'))
        if key == 'i':
            self.set('all_invisibility', not self.get('all_invisibility'))
        if key == 'j':
            self.set('hide_background', not self.get('hide_background'))
        if key == 'b':
            self.set('black', not self.get('black'))
        if key == 'g':
            self.set('gray_game', not self.get('gray_game'))
            self.id_position_map = {}
        if key == 's':
            self.set('show_boxes', not self.get('show_boxes'))
        if key == 'p':
            self.set('show_poses', not self.get('show_poses'))

        if key == '1':
            self.set('segmentation_parts', 0)
            self.set('segmentation_change', True)
        if key == '2':
            self.set('segmentation_parts', 1)
            self.set('segmentation_change', True)
        if key == '3':
            self.set('segmentation_parts', 2)
            self.set('segmentation_change', True)
        if key == '4':
            self.set('segmentation_parts', 3)
            self.set('segmentation_change', True)
        if key == '5':
            self.set('segmentation_parts', 4)
            self.set('segmentation_change', True)

        if key == 'o':
            self.set('overall_mirror', True)
            self.set('random_people_mirror', False)
            self.set('form_invisibility', False)
            self.set('all_invisibility', False)
            self.set('hide_background', False)
            self.set('gray_game', False)
            self.set('black', False)
            self.set('show_boxes', False)
            self.set('show_poses', False)
            self.id_position_map = {}
            self.set('segmentation_parts', 0)
            self.set('segmentation_change', True)

    def check_segmentation(self) -> Tuple[bool, int]:
        changed = bool(self.get('segmentation_change'))
        self.set('segmentation_change', False)
        return changed, int(self.get('segmentation_parts'))

    def copy(self, settings: Optional[GameSettings] = None) -> GameSettings:
        settings = GameSettings() if settings is None else settings
        for key, value in self.__dict__.items():
            setattr(settings, key, value)
        return settings

    def print(self) -> None:
        print(
            'Controls:'
            + '\nPress "f" for hiding people based on their area covering the image or pose data.'  # noqa: E501
            + '\nPress "i" for triggering invisibility of all detected people.'
            + '\nPress "j" for hiding everything except all detected people.'
            + '\nPress "g" for modifying color of all detected people based on their position.'  # noqa: E501
            + '\nPress "r" for randomly mirror position and mask of some people.'  # noqa: E501
            + '\nPress "b" for triggering a black screen.'
            + '\nPress "m" for mirroring the image.'
            + '\nPress "s" for rendering bounding boxes of detected people.'
            + '\nPress "p" for rendering poses of detected people.'
            + '\nPress "o" for resetting all settings.'
        )


class SharedSettingsManager:
    def __init__(self) -> None:
        BaseManager.register('GameSettings', GameSettings)
        self.manager = BaseManager()
        self.manager.start()

    def shared_settings(self, game_settings: GameSettings) -> GameSettings:
        shared_game_settings = self.manager.GameSettings()  # type: ignore
        shared_game_settings = game_settings.copy(shared_game_settings)
        return shared_game_settings

    def close(self) -> None:
        self.manager.shutdown()
