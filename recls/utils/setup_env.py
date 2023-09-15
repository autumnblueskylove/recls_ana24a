# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmpretrain into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmpretrain default
            scope. If True, the global default scope will be set to
            `mmpretrain`, and all registries will build modules from
            mmpretrain's registry node. To understand more about the registry,
            please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa: E501
    import recls.datasets  # noqa F401,F403
    import recls.engine  # noqa F401,F403
    import recls.evaluation  # noqa F401,F403
    import recls.models  # noqa: F401,F403
    import recls.visualization  # noqa: F401,F403

    if not init_default_scope:
        return

    current_scope = DefaultScope.get_current_instance()
    if current_scope is None:
        DefaultScope.get_instance('recls', scope_name='recls')
    elif current_scope.scope_name != 'recls':
        warnings.warn(
            f'The current default scope "{current_scope.scope_name}" '
            'is not "recls", `register_all_modules` will force '
            'the current default scope to be "recls". If this is '
            'not expected, please set `init_default_scope=False`.')
        # avoid name conflict
        new_instance_name = f'recls-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='recls')
