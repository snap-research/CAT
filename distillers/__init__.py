"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import importlib


def find_distiller_using_name(distiller_name):
    distiller_filename = "distillers." + distiller_name + '_distiller'
    modellib = importlib.import_module(distiller_filename)
    distiller = None
    target_distiller_name = distiller_name.replace('_', '') + 'distiller'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_distiller_name.lower():
            distiller = cls

    if distiller is None:
        print(
            "In %s.py, there should be a class of Distiller with class name that matches %s in lowercase."
            % (distiller_filename, target_distiller_name))
        exit(0)

    return distiller


def get_option_setter(distiller_name):
    distiller_class = find_distiller_using_name(distiller_name)
    return distiller_class.modify_commandline_options


def create_distiller(opt, verbose=True):
    distiller = find_distiller_using_name(opt.distiller)
    instance = distiller(opt)
    if verbose:
        print("distiller [%s] was created" % type(instance).__name__)
    return instance
