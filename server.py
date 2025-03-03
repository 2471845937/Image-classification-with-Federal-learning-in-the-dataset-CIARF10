import flwr as fl
import torch
import numpy as np
from torch import nn
from typing import List, Tuple, Dict
from data_loader import get_testloader
from model import MobileNetV2

torch.backends.cudnn.benchmark=True
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """加权平均指标聚合（兼容键缺失）"""
    losses = []
    accuracies = []
    total_examples = 0

    for num_examples, m in metrics:
        if "loss" in m:
            losses.append(num_examples * m["loss"])
        if "accuracy" in m:
            accuracies.append(num_examples * m["accuracy"])
        total_examples += num_examples

    avg_loss = sum(losses) / total_examples if losses else 0.0
    avg_accuracy = sum(accuracies) / total_examples if accuracies else 0.0

    return {"avg_loss": avg_loss, "avg_accuracy": avg_accuracy}


class FedProxStrategy(fl.server.strategy.FedAvg):
    def __init__(self, testloader, **kwargs):
        super().__init__(
            fit_metrics_aggregation_fn=weighted_average,
            **kwargs
        )
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_params = None  # 保存全局参数（Parameters 对象）

    def aggregate_fit(self, server_round, results, failures):
        """聚合训练参数并保存"""
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.global_params = aggregated_parameters  # 保存为 Parameters 对象
        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """全局测试集评估"""
        if not results or self.global_params is None:
            return None

        # 使用 Flower 工具函数将 Parameters 转换为 NumPy 数组
        from flwr.common import parameters_to_ndarrays
        parameters = parameters_to_ndarrays(self.global_params)

        # 加载模型
        self.model = MobileNetV2().to(self.device)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(np.copy(v)).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 计算全局准确率
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                correct += (torch.argmax(outputs, dim=1) == y).sum().item()
                total += y.size(0)
        accuracy = 100.0 * correct / total
        print(f"\n[Round {server_round}] Global Test Accuracy: {accuracy:.2f}%")

        return 0.0, {"accuracy": accuracy}


if __name__ == "__main__":
    testloader = get_testloader()
    strategy = FedProxStrategy(
        testloader=testloader,
        fraction_fit=1.0,
        min_available_clients=2,
        min_fit_clients=2,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )