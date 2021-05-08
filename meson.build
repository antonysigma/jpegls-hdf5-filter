project('jpegls-hdf-filter', 'cpp',
    version: '0.1',
    license: 'MIT',
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

hdf5_dep = dependency('hdf5-serial')
threads_dep = dependency('threads')

h5jpegls_lib = library('h5jpegls',
    sources: [
        'h5jpegls.cc',
    ],
    include_directories: [
        charls_inc,
    ],
    cpp_args: [
        '-DNO_DEBUG',
    ],
    link_with: [
        charls_lib,
    ],
    dependencies: [
        hdf5_dep,
        threads_dep,
    ],
)
