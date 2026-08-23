import argparse
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
    parser.add_argument(
        "--d0-checkpoint",
        type=str,
        required=True,
        help="Matched D0FT seed checkpoint used as the only initialization source.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/af2_orient_cmc0/AF2_ORIENT_CMC0_yolo26n.yaml",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    d0_checkpoint = Path(args.d0_checkpoint).expanduser().resolve()

    if not (data_root / "data.yaml").exists():
        raise FileNotFoundError(f"Missing grouped dataset yaml: {data_root / 'data.yaml'}")
    if not d0_checkpoint.exists():
        raise FileNotFoundError(f"Missing D0FT checkpoint: {d0_checkpoint}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    trainer_cls = make_af2_orient_cmc0_trainer(
        af2_config=cfg.get("af2_iso", {}),
        stb_config=cfg.get("stb", {}),
        d0_checkpoint=d0_checkpoint,
    )

    # The trainer constructs AF2_ORIENT+CMC0 and strictly transfers the matched
    # D0FT checkpoint into the native detector state. Do not initialize the
    # experiment scientifically from generic yolo26n.pt.
    model = YOLO("yolo26n.pt")
    model.TrainerClass = trainer_cls

    model.train(
        data=str(data_root / "data.yaml"),
        project=str(output_root),
        name=f"AF2_ORIENT_CMC0_seed{args.seed}",
        seed=args.seed,
        device=args.device,
        **cfg.get("train", {}),
    )


if __name__ == "__main__":
    main()
