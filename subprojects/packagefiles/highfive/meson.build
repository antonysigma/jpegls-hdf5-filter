project('highfive', 'cpp', 'c',
    version: 'v1.10.4-6',
)

# Header-only library
highfive_inc = include_directories('include')

# highfive is a high-level abstraction of the
# C-functions in the libhdf5 library 
highfive_dep = declare_dependency(
    include_directories: highfive_inc,
    dependencies: [
        dependency('hdf5', language: 'c'),
    ]
)

read_write_vector_exe = executable(
    'read_write_vector',
    sources: 'src/examples/read_write_vector_dataset.cpp',
    dependencies: highfive_dep,
)

read_write_partial_exe = executable(
    'select_partial_dataset_cpp11',
    sources: 'src/examples/select_partial_dataset_cpp11.cpp',
    dependencies: highfive_dep,
)

# Test the highfive APIs

test('read and write vector',
    read_write_vector_exe,
    suite: 'highfive',
)

test('read only a few pixels',
    read_write_partial_exe,
    suite: 'highfive',
)