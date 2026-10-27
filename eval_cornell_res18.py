"""ResNet-18 (including structured-pruned) Cornell holdout evaluation."""
from eval_cornell import CornellEvaluator
from grasp_evaluation import evaluation_cli
from train_resnet_18 import get_model


class CornellEvaluatorRes18(CornellEvaluator):
    def __init__(self, model_path, cornell_path, device='auto', **kwargs):
        super().__init__(model_path, cornell_path, device, model_builder=get_model, **kwargs)


if __name__ == '__main__':
    evaluation_cli(CornellEvaluatorRes18, 'cornell')
