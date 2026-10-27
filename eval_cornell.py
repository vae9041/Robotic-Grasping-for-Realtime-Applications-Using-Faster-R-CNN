"""ResNet-50 Cornell evaluation with true rotated geometry and a saved holdout."""
from grasp_evaluation import GraspEvaluator, evaluation_cli
from train import get_model


class CornellEvaluator(GraspEvaluator):
    def __init__(self, model_path, cornell_path, device='auto', model_builder=None, **kwargs):
        super().__init__(model_path, cornell_path, 'original_cornell', model_builder or get_model,
                         device=device, **kwargs)


def main():
    evaluation_cli(CornellEvaluator, 'cornell')


if __name__ == '__main__':
    main()
