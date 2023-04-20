import os

multimodal_backbones = {}
def register_multimodal_backbone(name):
    def decorator(cls):
        multimodal_backbones[name] = cls
        return cls
    return decorator

dependency_blocks = {}
def register_dependency_block(name):
    def decorator(cls):
        dependency_blocks[name] = cls
        return cls
    return decorator

multimodal_meta_archs = {}
def register_multimodal_meta_arch(name):
    def decorator(cls):
        multimodal_meta_archs[name] = cls
        return cls
    return decorator

# builder functions
def make_multimodal_backbone(name, **kwargs):
    multimodal_backbone = multimodal_backbones[name](**kwargs)
    return multimodal_backbone

def make_dependency_block(name, **kwargs):
    dependency_block = dependency_blocks[name](**kwargs)
    return dependency_block

def make_multimodal_meta_arch(name, **kwargs):
    multimodal_meta_arch = multimodal_meta_archs[name](**kwargs)
    return multimodal_meta_arch

