highfive_dep = subproject('highfive').get_variable('highfive_dep')

cxx = meson.get_compiler('cpp')
hdf5_hl_lib = cxx.find_library('hdf5_hl', dirs: '/usr/lib/x86_64-linux-gnu/hdf5/serial')

sync_write_exe = executable('sync_write',
    sources: [
        'sync-write.cpp',
    ],
    dependencies: [
        highfive_dep,
    ],
)

direct_chunk_write_exe = executable('direct_chunk_write',
    sources: [
        'direct_chunk_write.cpp',
    ],
    dependencies: [
        highfive_dep,
        jpegls_filter_async_dep,
        hdf5_hl_lib,
    ],
)

test('Write using dynamic plugin',
    sync_write_exe,
    env: {
        'HDF5_PLUGIN_PATH': meson.current_build_dir() / '..',
    },
    suite: 'unittest',
)

test('Writing small chunks',
    direct_chunk_write_exe,
    env: {
        'HDF5_PLUGIN_PATH': meson.current_build_dir() / '..',
    },
    suite: 'unittest',
    is_parallel: false,
)
