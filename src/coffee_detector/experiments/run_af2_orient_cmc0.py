import argparse
import sys
from pathlib import Path
import yaml

from coffee_detector.af2_orient_cmc0.trainer import make_af2_orient_cmc0_trainer
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--config", type=str, default="configs/af2_orient_cmc0/AF2_ORIENT_CMC0_yolo26n.yaml")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.config).resolve()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    trainer_cls = make_af2_orient_cmc0_trainer(
        af2_config=cfg.get("af2_iso", {}),
        stb_config=cfg.get("stb", {}),
        d0_checkpoint=None
    )

    model = YOLO("yolo26n.pt")
    # Ganti trainer default dengan trainer custom kita
    model.TrainerClass = trainer_cls

    model.train(
        data=str(data_root / "data.yaml"),
        project=str(output_root),
        name=f"AF2_ORIENT_CMC0_seed{args.seed}",
        seed=args.seed,
        device=args.device,
        **cfg.get("train", {})
    )

if __name__ == "__main__":
    main()
