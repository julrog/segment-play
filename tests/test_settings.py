import random
from multiprocessing import Process

from pytest import CaptureFixture

from src.settings import GameSettings, SharedSettingsManager


def test_handle_key() -> None:
    settings = GameSettings()

    # Test toggling overall_mirror
    settings.handle_key('m')
    assert not settings.get('overall_mirror')
    settings.handle_key('m')
    assert settings.get('overall_mirror')

    # Test toggling random_people_mirror
    settings.handle_key('r')
    assert settings.get('random_people_mirror')
    settings.handle_key('r')
    assert not settings.get('random_people_mirror')

    # Test toggling form_invisibility
    settings.handle_key('f')
    assert settings.get('form_invisibility')
    settings.handle_key('f')
    assert not settings.get('form_invisibility')

    # Test toggling all_invisibility
    settings.handle_key('i')
    assert settings.get('all_invisibility')
    settings.handle_key('i')
    assert not settings.get('all_invisibility')

    # Test toggling hide_background
    settings.handle_key('j')
    assert settings.get('hide_background')
    settings.handle_key('j')
    assert not settings.get('hide_background')

    # Test toggling black
    settings.handle_key('b')
    assert settings.get('black')
    settings.handle_key('b')
    assert not settings.get('black')

    # Test toggling gray_game
    settings.handle_key('g')
    assert settings.get('gray_game')
    assert settings.id_position_map == {}
    settings.handle_key('g')
    assert not settings.get('gray_game')

    # Test toggling show_boxes
    settings.handle_key('s')
    assert settings.get('show_boxes')
    settings.handle_key('s')
    assert not settings.get('show_boxes')

    # Test toggling show_poses
    settings.handle_key('p')
    assert settings.get('show_poses')
    settings.handle_key('p')
    assert not settings.get('show_poses')

    # Test changing segmentation_parts
    settings.handle_key('1')
    assert settings.get('segmentation_parts') == 0
    assert settings.get('segmentation_change')
    settings.handle_key('2')
    assert settings.get('segmentation_parts') == 1
    assert settings.get('segmentation_change')
    settings.handle_key('3')
    assert settings.get('segmentation_parts') == 2
    assert settings.get('segmentation_change')
    settings.handle_key('4')
    assert settings.get('segmentation_parts') == 3
    assert settings.get('segmentation_change')
    settings.handle_key('5')
    assert settings.get('segmentation_parts') == 4
    assert settings.get('segmentation_change')

    # Test resetting all settings
    settings.handle_key('o')
    assert settings.get('overall_mirror')
    assert not settings.get('random_people_mirror')
    assert not settings.get('form_invisibility')
    assert not settings.get('all_invisibility')
    assert not settings.get('hide_background')
    assert not settings.get('gray_game')
    assert not settings.get('black')
    assert not settings.get('show_boxes')
    assert not settings.get('show_poses')
    assert settings.id_position_map == {}
    assert settings.get('segmentation_parts') == 0
    assert settings.get('segmentation_change')


def test_check_segmentation() -> None:
    settings = GameSettings()
    settings.set('segmentation_change', True)
    settings.set('segmentation_parts', 2)
    assert settings.check_segmentation() == (True, 2)
    assert settings.check_segmentation() == (False, 2)


def test_copy_to() -> None:
    settings = GameSettings()
    settings.set('overall_mirror', random.choice([True, False]))
    settings.set('random_people_mirror', random.choice([True, False]))
    settings.set('form_invisibility', random.choice([True, False]))
    settings.set('all_invisibility', random.choice([True, False]))
    settings.set('hide_background', random.choice([True, False]))
    settings.set('gray_game', random.choice([True, False]))
    settings.set('black', random.choice([True, False]))
    settings.set('show_boxes', random.choice([True, False]))
    settings.set('show_poses', random.choice([True, False]))
    settings.id_position_map = {}
    settings.set('segmentation_parts', random.choice([0, 1]))
    settings.set('segmentation_change', random.choice([True, False]))

    new_settings = GameSettings()
    settings.copy_to(new_settings)

    assert new_settings.get('overall_mirror') == settings.get('overall_mirror')
    assert new_settings.get('random_people_mirror') == settings.get(
        'random_people_mirror')
    assert new_settings.get(
        'form_invisibility') == settings.get('form_invisibility')
    assert new_settings.get(
        'all_invisibility') == settings.get('all_invisibility')
    assert new_settings.get(
        'hide_background') == settings.get('hide_background')
    assert new_settings.get('gray_game') == settings.get('gray_game')
    assert new_settings.get('black') == settings.get('black')
    assert new_settings.get('show_boxes') == settings.get('show_boxes')
    assert new_settings.get('show_poses') == settings.get('show_poses')
    assert new_settings.id_position_map == settings.id_position_map
    assert new_settings.get('segmentation_parts') == settings.get(
        'segmentation_parts')
    assert new_settings.get('segmentation_change') == settings.get(
        'segmentation_change')


def test_print(capsys: CaptureFixture) -> None:
    settings = GameSettings()
    settings.print()
    captured = capsys.readouterr()
    assert captured.out.startswith('Controls:')


def change_setting_value(game_settings: GameSettings) -> None:
    game_settings.set('save_imgs', True)
    game_settings.update_position_map(5, 0.5)
    game_settings.set('segmentation_parts',
                      game_settings.get('segmentation_parts') + 1)
    if game_settings.get('form_invisibility'):
        game_settings.set('form_invisibility', False)
    else:
        game_settings.set('form_invisibility', True)


def test_shared_settings() -> None:
    settings = GameSettings()
    manager = SharedSettingsManager()
    shared_settings = manager.shared_settings(settings)

    assert not shared_settings.get('form_invisibility')
    shared_settings.set('segmentation_parts',
                        shared_settings.get('segmentation_parts') + 1)

    p = Process(target=change_setting_value, args=[shared_settings])
    p.start()
    p.join()

    assert shared_settings.get('segmentation_parts') == 3
    assert not settings.get('save_imgs')
    assert shared_settings.get('save_imgs')
    assert not settings.has_position_map(5)
    assert shared_settings.has_position_map(5)
    assert shared_settings.get_position_map(5) == 0.5
    assert shared_settings.get('form_invisibility')

    manager.close()
