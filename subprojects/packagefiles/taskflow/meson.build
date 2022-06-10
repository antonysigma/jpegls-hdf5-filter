project('taskflow', 'cpp',
    version: 'v2.7.0',
)

# Header-only library
taskflow_inc = include_directories('.')

taskflow_dep = declare_dependency(
    include_directories: taskflow_inc,
    dependencies: [
        dependency('threads'),
    ]
)