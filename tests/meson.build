cxx = meson.get_compiler('cpp')
if cxx.has_argument('-fsanitize=fuzzer')

fuzzer_exe = executable('fuzzer-main',
    sources: 'fuzzer-main.cpp',
    cpp_args: '-fsanitize=fuzzer-no-link',
    link_args: '-fsanitize=fuzzer',
    dependencies: highfive_dep,
)

test('Monkey test',
    fuzzer_exe,
    args: [
        '-max_total_time=10',
        '-max_len=4',
        '-workers=1',
    ],
    env: {
        'HDF5_PLUGIN_PATH': meson.current_build_dir() / '..',
    },
    suite: 'fuzzer',
    is_parallel: false,
)

endif