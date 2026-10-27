"""ResNet-50 Jacquard evaluation with the same preprocessing and metric as training."""
from grasp_evaluation import GraspEvaluator, evaluation_cli
from train import get_model


class JacquardEvaluator(GraspEvaluator):
    def __init__(self, model_path, jacquard_path, device='auto', model_builder=None, **kwargs):
        super().__init__(model_path, jacquard_path, 'original_jacquard', model_builder or get_model,
                         device=device, **kwargs)

    def evaluate(self, **kwargs):
        return self.run(**kwargs)[0]


def main():
    evaluation_cli(JacquardEvaluator, 'jacquard')


if __name__ == '__main__':
    main()
