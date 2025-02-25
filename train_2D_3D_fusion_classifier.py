import argparse
from openpoints.utils import EasyConfig




if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training a BioVista dataset classifier consisting of a PointVector and a ResNet Model which are then fused together with a MLP for fusion')
    parser.add_argument('--pointvector_cfg', type=str, help='config file',
                        default="cfgs/biovista/pointvector-s.yaml")
    


    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.pointvector_cfg, recursive=True)
    cfg.update(opts)


