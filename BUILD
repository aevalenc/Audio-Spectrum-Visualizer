load("@rules_python//python:defs.bzl", "py_binary")

# load("@aspect_rules_py//py:defs.bzl", "py_binary")
load("@rules_python//python/pip_install:requirements.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
    visibility = ["//visibility:public"],
)

exports_files(["requirements_lock.txt"])

py_binary(
    name = "audio_visualizer",
    srcs = ["audio_visualizer.py"],
    main = "audio_visualizer.py",
    # package_collisions = "warning",
    deps = [
        "@pypi//numpy",
        "@pypi//pyqt5",
        "@pypi//pyqtgraph",
        "@pypi//scipy",
        "@pypi//sounddevice",
    ],
)
