from .vit import vit, vit_base

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base':
        return vit_base(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
