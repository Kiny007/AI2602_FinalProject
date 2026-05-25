import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.metrics import inception_score
from gan_faces.utils import get_device, load_generator_from_checkpoint, save_json, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Inception Score 评估生成头像质量")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="outputs/metrics/is_score.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    generator, model_args, _ = load_generator_from_checkpoint(args.checkpoint, device)
    latent_dim = int(model_args.get("latent_dim", 100))

    mean, std = inception_score(
        generator=generator,
        latent_dim=latent_dim,
        num_images=args.num_images,
        batch_size=args.batch_size,
        splits=args.splits,
        device=device,
    )

    result = {
        "metric": "Inception Score",
        "mean": mean,
        "std": std,
        "num_images": args.num_images,
        "splits": args.splits,
        "checkpoint": args.checkpoint,
    }
    save_json(result, args.output_json)
    print(f"Inception Score: mean={mean:.4f}, std={std:.4f}")
    print(f"评估结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
