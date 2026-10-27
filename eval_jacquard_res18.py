"""ResNet-18 (including structured-pruned) Jacquard holdout evaluation."""
from eval_jacquard import JacquardEvaluator
from grasp_evaluation import evaluation_cli
from train_resnet_18 import get_model


class JacquardEvaluatorRes18(JacquardEvaluator):
    def __init__(self, model_path, jacquard_path, device='auto', **kwargs):
        super().__init__(model_path, jacquard_path, device, model_builder=get_model, **kwargs)


if __name__ == '__main__':
    evaluation_cli(JacquardEvaluatorRes18, 'jacquard')
