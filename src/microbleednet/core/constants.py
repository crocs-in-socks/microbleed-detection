from pathlib import Path

class transforms:
    class frangi:
        sigmas = (0.5, 1.2, 0.2)
        alpha = 0.9
        beta = 20
        black_ridges = False

    class vessel_inpainting:
        clusterer_n_clusters = 2
        clusterer_random_state = 42
        minimum_vessel_eccentricity = 0.9
        maximum_vessel_solidity = 0.5

    class frst:
        radii = [2, 3, 4, 6]
        alpha = 2
        factor_std = 0.1
        bright = True
        dark = False


class engines:
    class trainers:
        best_checkpoint_path = Path("best_model.pth")
        latest_checkpoint_path = Path("latest_model.pth")

        class default:
            compile_model = True
            clip_norm = 1.0
            checkpoint_path = None
            weights_only = False


class dataloading:
    class patchers:
        class default:
            augmentation_factor = 1


class common:
    class layers:
        class single_conv:
            class default:
                padding = 1

    class losses:
        class dice:
            class default:
                smooth = 1.0
