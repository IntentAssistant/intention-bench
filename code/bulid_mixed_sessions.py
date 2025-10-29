import os
import json
import random
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

WORK_PATH = REPO_ROOT

DEFAULT_TRAJECTORY_PATH = WORK_PATH / "intention_bench" / "dataset" / "images"
if not DEFAULT_TRAJECTORY_PATH.is_dir():
    DEFAULT_TRAJECTORY_PATH = WORK_PATH / "intention_bench" / "dataset" / "focused_session_data"
if not DEFAULT_TRAJECTORY_PATH.is_dir():
    DEFAULT_TRAJECTORY_PATH = WORK_PATH / "dataset" / "trajectory_data"

DEFAULT_OUTPUT_PATH = (
    WORK_PATH
    / "intention_bench"
    / "dataset"
    / "annotations"
    / "mixed_sessions"
    / "raw_jsons"
)


def synthesize_data(
    trajectories: dict,
    data_type: int = None,
    base_trajectory: str = None,
    target_trajectories: list = None,
) -> dict:
    """
    data_type 0: single trajectory
    data_type 1: mixed with other categories (base + target from different category)
    data_type 2: mixed within same category (base + target from same category, excluding base)
    Returns: dictionary matching the mixed-session JSON schema or None if sampling fails
    """
    if data_type == 0:
        # Single trajectory - use base_trajectory if provided, otherwise random
        if base_trajectory:
            trajectory_0 = base_trajectory
        else:
            trajectory_0_categories = list(
                set([code_name[:4] for code_name in trajectories.keys()])
            )
            trajectory_0_category = random.choice(trajectory_0_categories)
            trajectory_0 = random.choice(
                [key for key in trajectories.keys() if key[:4] == trajectory_0_category]
            )

        trajectory_1 = None

        subset = trajectories[trajectory_0]
        shuffled_paths = []
        labels = []
        for key in subset:
            paths = subset[key]
            shuffled_paths.extend(paths)
            labels.extend([0] * len(paths))

    elif data_type == 1:
        # Mixed with other categories: base + target from different category
        if base_trajectory:
            trajectory_0 = base_trajectory
        else:
            trajectory_0 = random.choice(list(trajectories.keys()))

        if target_trajectories:
            trajectory_1 = target_trajectories[0]
        else:
            candidates = [
                key for key in trajectories.keys() if key[:4] != trajectory_0[:4]
            ]
            if not candidates:
                return None
            trajectory_1 = random.choice(candidates)

        all_base_keys = list(trajectories[trajectory_0].keys())
        all_target_keys = list(trajectories[trajectory_1].keys())

        if not all_base_keys or not all_target_keys:
            return None

        selected_base_keys = random.sample(
            all_base_keys, random.randint(1, len(all_base_keys))
        )
        selected_target_keys = random.sample(
            all_target_keys, random.randint(1, len(all_target_keys))
        )

        subset_a = {key: trajectories[trajectory_0][key] for key in selected_base_keys}
        subset_b = {
            key: trajectories[trajectory_1][key] for key in selected_target_keys
        }

        shuffle_order = [(key, 0) for key in subset_a] + [(key, 1) for key in subset_b]
        random.shuffle(shuffle_order)

        shuffled_paths = []
        labels = []
        for key, label in shuffle_order:
            paths = subset_a[key] if label == 0 else subset_b[key]
            shuffled_paths.extend(paths)
            labels.extend([label] * len(paths))

    elif data_type == 2:
        # Mixed within same category: base + 1 target from same category
        if base_trajectory:
            trajectory_0 = base_trajectory
        else:
            trajectory_0 = random.choice(list(trajectories.keys()))

        if target_trajectories:
            trajectory_1 = target_trajectories[0]
        else:
            candidates = [
                key
                for key in trajectories.keys()
                if key[:4] == trajectory_0[:4] and key != trajectory_0
            ]
            if not candidates:
                return None
            trajectory_1 = random.choice(candidates)

        all_base_keys = list(trajectories[trajectory_0].keys())
        all_target_keys = list(trajectories[trajectory_1].keys())

        if not all_base_keys or not all_target_keys:
            return None

        selected_base_keys = random.sample(
            all_base_keys, random.randint(1, len(all_base_keys))
        )
        selected_target_keys = random.sample(
            all_target_keys, random.randint(1, len(all_target_keys))
        )

        subset_a = {key: trajectories[trajectory_0][key] for key in selected_base_keys}
        subset_b = {
            key: trajectories[trajectory_1][key] for key in selected_target_keys
        }

        shuffle_order = [(key, 0) for key in subset_a] + [(key, 1) for key in subset_b]
        random.shuffle(shuffle_order)

        shuffled_paths = []
        labels = []
        for key, label in shuffle_order:
            paths = subset_a[key] if label == 0 else subset_b[key]
            shuffled_paths.extend(paths)
            labels.extend([label] * len(paths))

    # Build return data
    result = {
        "trajectory_0": trajectory_0,
        "trajectories": shuffled_paths,
        "labels": labels,
    }

    if data_type == 1:
        result["trajectory_1"] = trajectory_1
    elif data_type == 2:
        result["trajectory_1"] = trajectory_1
    else:
        result["trajectory_1"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Synthesize trajectory data")
    parser.add_argument(
        "--data_mode",
        choices=["train", "test", "all"],
        default="train",
        help="Data mode: 'train' (1-8), 'test' (9-10), 'all' (1-10)",
    )
    parser.add_argument(
        "--simple_mode",
        action="store_true",
        help="Simple mode: only Type 0 (single) + Type 1 (cross-category), skip Type 2 (same-category)",
    )
    parser.add_argument(
        "--cross_only",
        action="store_true",
        help="Generate only Type 1 (cross-category) data with random mixing",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=str(DEFAULT_TRAJECTORY_PATH),
        help=(
            "Directory containing focused session trajectories "
            f"(default: {DEFAULT_TRAJECTORY_PATH})"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Directory to store synthesized mixed sessions "
            f"(default: {DEFAULT_OUTPUT_PATH})"
        ),
    )
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory_dir).expanduser().resolve()
    if not trajectory_path.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_path}")

    output_path = Path(args.output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Focused sessions source: {trajectory_path}")
    print(f"💾 Mixed sessions output: {output_path}")

    # Set random seed for reproducibility
    random.seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")

    if args.simple_mode:
        print("🔧 Simple mode enabled: Type 0 (Single) + Type 1 (Cross-category) only")

    if args.cross_only:
        print(
            "🔧 Cross-only mode enabled: Type 1 (Cross-category) only with random ratios"
        )

    # Load trajectories
    training_trajectories = defaultdict(dict)
    test_trajectories = defaultdict(dict)

    for json_file in glob(str(trajectory_path / "*" / "_subtrajectory_data.json")):
        with open(json_file) as f:
            data = json.load(f)

        code_name = data["code_name"]  # e.g., SHOP_001
        trajectory_id = int(code_name.split("_")[1])  # e.g., 1 from SHOP_001

        for key, items in data["sub_trajectories"].items():
            file_paths = [item for item in items]

            if 1 <= trajectory_id <= 8:
                training_trajectories[code_name][key] = file_paths
            elif 9 <= trajectory_id <= 10:
                test_trajectories[code_name][key] = file_paths

    print(f"# of training trajectories: {len(training_trajectories)}")
    print(f"# of test trajectories: {len(test_trajectories)}")

    # Select trajectories based on data_mode
    if args.data_mode == "train":
        selected_trajectories = training_trajectories
        if args.cross_only:
            num_samples = args.num_samples
            print(
                f"🔧 Using TRAINING trajectories only (1-8) → {num_samples} cross-category samples"
            )
        else:
            num_samples = 40
            print("🔧 Using TRAINING trajectories only (1-8) → 40 samples each")
    elif args.data_mode == "test":
        selected_trajectories = test_trajectories
        if args.cross_only:
            num_samples = args.num_samples
            print(
                f"🧪 Using TEST trajectories only (9-10) → {num_samples} cross-category samples"
            )
        else:
            num_samples = 10
            print("🧪 Using TEST trajectories only (9-10) → 10 samples each")
    elif args.data_mode == "all":
        # Merge training and test trajectories
        selected_trajectories = defaultdict(dict)
        for traj_name, subtrajs in training_trajectories.items():
            selected_trajectories[traj_name] = subtrajs
        for traj_name, subtrajs in test_trajectories.items():
            selected_trajectories[traj_name] = subtrajs
        if args.cross_only:
            num_samples = args.num_samples
            print(
                f"🔄 Using ALL trajectories (1-10) → {num_samples} cross-category samples"
            )
        else:
            num_samples = 50
            print("🔄 Using ALL trajectories (1-10) → 50 samples each")

    # Get all categories and trajectories by category
    all_trajectories = list(selected_trajectories.keys())
    categories = {}
    for traj in all_trajectories:
        category = traj[:4]
        if category not in categories:
            categories[category] = []
        categories[category].append(traj)

    print(f"📊 Found categories: {list(categories.keys())}")
    for cat, trajs in categories.items():
        print(f"  {cat}: {len(trajs)} trajectories")

    # Synthesize data
    file_counter = 0
    type0_counter = 0
    type1_counter = 0
    type2_counter = 0

    if args.cross_only:
        # Generate ONLY Type 1 (Cross-category) with specified number of samples
        print(
            f"\n🔄 Generating Type 1 (Cross-category) ONLY - {num_samples} samples with random ratios"
        )
        generated_type_1 = 0

        # Continue generating until we have the desired number of samples
        while generated_type_1 < num_samples:
            # Pick random base trajectory
            base_traj = random.choice(all_trajectories)
            base_category = base_traj[:4]
            other_categories = [
                cat for cat in categories.keys() if cat != base_category
            ]

            if other_categories:  # Make sure there are other categories
                # Pick random target category and trajectory
                target_cat = random.choice(other_categories)
                target_traj = random.choice(categories[target_cat])

                data = synthesize_data(
                    selected_trajectories,
                    data_type=1,
                    base_trajectory=base_traj,
                    target_trajectories=[target_traj],
                )

                if not data:
                    continue

                with (output_path / f"type1_cross_{type1_counter:03d}.json").open(
                    "w"
                ) as f:
                    json.dump(data, f, indent=4)

                type1_counter += 1
                generated_type_1 += 1

        print(f"✅ Generated {generated_type_1} Type 1 samples with random ratios")
    else:
        # Original logic for multiple types
        # TYPE 0: Single trajectory (num_samples개)
        print(f"\n🔄 Generating Type 0 (Single) - {num_samples} samples")
        used_type_0 = set()
        while len(used_type_0) < num_samples:
            data = synthesize_data(selected_trajectories, data_type=0)

            if data["trajectory_0"] in used_type_0:
                continue

            with (output_path / f"type0_single_{type0_counter:03d}.json").open(
                "w"
            ) as f:
                json.dump(data, f, indent=4)

            used_type_0.add(data["trajectory_0"])
            type0_counter += 1

        print(f"✅ Generated {len(used_type_0)} Type 0 samples")

        # TYPE 1: Mixed with other categories (num_samples * 4개)
        print(f"\n🔄 Generating Type 1 (Cross-category) - {num_samples * 4} samples")
        generated_type_1 = 0

        # For each trajectory as base
        for base_traj in all_trajectories[:num_samples]:  # Limit base trajectories
            base_category = base_traj[:4]
            other_categories = [
                cat for cat in categories.keys() if cat != base_category
            ]

            # Get one trajectory from each other category
            for other_cat in other_categories:
                if other_cat in categories and categories[other_cat]:
                    target_traj = random.choice(categories[other_cat])

                data = synthesize_data(
                    selected_trajectories,
                    data_type=1,
                    base_trajectory=base_traj,
                    target_trajectories=[target_traj],
                )

                if not data:
                    continue

                with (output_path / f"type1_cross_{type1_counter:03d}.json").open(
                    "w"
                ) as f:
                    json.dump(data, f, indent=4)

                type1_counter += 1
                generated_type_1 += 1

        print(f"✅ Generated {generated_type_1} Type 1 samples")

        # TYPE 2: Same category mix (num_samples * 2개) - Skip in simple mode
        generated_type_2 = 0

        if not args.simple_mode:
            print(f"\n🔄 Generating Type 2 (Same-category) - {num_samples * 2} samples")

            # For each trajectory as base
            for base_traj in all_trajectories[:num_samples]:  # Limit base trajectories
                base_category = base_traj[:4]
                same_category_trajs = [
                    t for t in categories[base_category] if t != base_traj
                ]

                if len(same_category_trajs) >= 1:
                    # Select 1 random trajectory from same category (excluding base)
                    target_trajs = [random.choice(same_category_trajs)]

                    data = synthesize_data(
                        selected_trajectories,
                        data_type=2,
                        base_trajectory=base_traj,
                        target_trajectories=target_trajs,
                    )

                    if not data:
                        continue

                    with (output_path / f"type2_same_{type2_counter:03d}.json").open(
                        "w"
                    ) as f:
                        json.dump(data, f, indent=4)

                    type2_counter += 1
                    generated_type_2 += 1

                    # Generate second sample with different random target
                    if len(same_category_trajs) >= 1:
                        target_trajs_2 = [random.choice(same_category_trajs)]

                        data = synthesize_data(
                            selected_trajectories,
                            data_type=2,
                            base_trajectory=base_traj,
                            target_trajectories=target_trajs_2,
                        )

                        if not data:
                            continue

                        with (
                            output_path / f"type2_same_{type2_counter:03d}.json"
                        ).open("w") as f:
                            json.dump(data, f, indent=4)

                        type2_counter += 1
                        generated_type_2 += 1

            print(f"✅ Generated {generated_type_2} Type 2 samples")
        else:
            print(f"\n⏭️ Skipping Type 2 (Same-category) in simple mode")

    total_generated = type0_counter + type1_counter + type2_counter
    print(f"\n🎉 Total generated: {total_generated} samples")

    if args.cross_only:
        print(f"  📁 Type 1 (Cross) (random ratios): {type1_counter} files")
        print(f"🔧 Cross-only mode - Generated: {type1_counter} samples")
    else:
        print(f"  📁 Type 0 (Single): {type0_counter} files")
        print(f"  📁 Type 1 (Cross): {type1_counter} files")
        if not args.simple_mode:
            print(f"  📁 Type 2 (Same): {type2_counter} files")

        if args.simple_mode:
            expected_total = num_samples + (num_samples * 4)  # Type 0 + Type 1
            print(
                f"🔧 Simple mode - Expected: {expected_total} samples (Type 0: {num_samples} + Type 1: {num_samples * 4})"
            )


if __name__ == "__main__":
    main()
