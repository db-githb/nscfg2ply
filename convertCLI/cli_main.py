import argparse
from pathlib import Path
from convertPY.main.convert_main import Convert
def main():
    parser = argparse.ArgumentParser(prog="convert", description="convert ns model defined in config to ply")
    sub = parser.add_subparsers(dest="command", required=True)
    
    p_cull = sub.add_parser("convert", help="Cull Gaussians from 3DGS model using binary masks")
    p_cull.add_argument("--model-path", "-l", required=True,
                        help="path to 3DGS model's yaml configuration file")
    p_cull.add_argument("--output-dir", "-o", default=None,
                        help="Path to output directory")
    args = parser.parse_args()

    if args.command == "convert":
        dc = Convert(Path(args.model_path), output_dir=args.output_dir)
        dc.run_convert()

if __name__ == "__main__":
    main()