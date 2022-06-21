highfive_dep = subproject('highfive').get_variable('highfive_dep')

sync_write_exe = executable('sync_write',
    sources: [
        'sync-write.cpp',
    ],
    dependencies: [
        highfive_dep,
    ],
)

test('Write using dynamic plugin',
    sync_write_exe,
    env: {
        'HDF5_PLUGIN_PATH': meson.current_build_dir() / '..',
    },
    suite: 'unittest',
)
