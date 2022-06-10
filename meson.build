project('jpegls-hdf-filter', 'cpp', 'c',
    version: '0.1',
    license: 'MIT',
    meson_version: '>=0.57',
    default_options : [
        'buildtype=debugoptimized',
        'b_ndebug=if-release',
        'cpp_std=c++17',
])

charls_proj = subproject('charls',
    default_options: 'default_library=static'
)
charls_lib = charls_proj.get_variable('charls_lib')
charls_inc = charls_proj.get_variable('charls_inc')

hdf5_dep = dependency('hdf5', language: 'c')
threads_dep = dependency('threads')
openmp_dep = dependency('openmp')
taskflow_dep = subproject('taskflow').get_variable('taskflow_dep')

jpegls_filter_dep = declare_dependency(
    include_directories: [
        '.',
        charls_inc,
    ],
    sources: [
        'jpegls-filter.cpp',
    ],
    link_with: [
        charls_lib,
    ],
    dependencies: [
        threads_dep,
        openmp_dep,
    ],
)

jpegls_filter_async_dep = declare_dependency(
    compile_args: '-DH5JPEGLS_USE_ASYNC',
    dependencies: [
        jpegls_filter_dep,
        taskflow_dep,
    ],
)

h5jpegls_lib = library('h5jpegls',
    sources: [
        'h5jpegls.cpp',
    ],
    include_directories: [
        charls_inc,
    ],
    cpp_args: [
        '-fvisibility=hidden',
        '-DNO_DEBUG',
    ],
    dependencies: [
        jpegls_filter_dep,
        hdf5_dep,
        threads_dep,
    ],
)

h5repack_exe = find_program('h5repack')
test_data = files('test-vector/bloated.hdf5')

compressed_data = custom_target('compressed.hdf5',
    input: [
        test_data,
        h5jpegls_lib,
    ],
    output: 'compressed.hdf5',
    env: {
      'HDF5_PLUGIN_PATH': meson.current_build_dir(),
    },
    command: [
        h5repack_exe,
        '-f', 'ones:UD=32012,3,0,0,0',
        '@INPUT0@',
        '@OUTPUT@',
    ],
    build_by_default: false,
)

restored_data = custom_target('restored.hdf5',
    input: [
        compressed_data,
        h5jpegls_lib,
    ],
    output: 'restored.hdf5',
    env: {
      'HDF5_PLUGIN_PATH': meson.current_build_dir(),
    },
    command: [
        h5repack_exe,
        '-f', 'ones:NONE',
        '@INPUT0@',
        '@OUTPUT@',
    ],
    build_by_default: false,
)

h5ls_exe = find_program('h5ls')
test('JPEG-LS compression w/ h5repack',
    h5ls_exe,
    args: [
        '-v',
        compressed_data,
    ],
)

test('JPEG-LS decoding w/ h5repack',
    h5ls_exe,
    args: [
        '-v',
        restored_data,
    ],
)

subdir('examples')
