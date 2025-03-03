import argparse
import flwr as fl
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from data_loader import non_iid_split, get_testloader
from model import MobileNetV2

torch.backends.cudnn.benchmark=True
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = MobileNetV2().to(self.device)

        # 加载数据
        client_datasets = non_iid_split(num_clients=4)
        self.trainloader = DataLoader(
            client_datasets[client_id],
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        self.testloader = get_testloader()

    def get_parameters(self, config):
        """返回模型参数"""
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(val).to(self.device) for k, val in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """本地训练"""
        self.set_parameters(parameters)

        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        mu = config.get("mu", 0.1)

        total_loss, total_correct, total_samples = 0.0, 0, 0

        self.model.train()
        for epoch in range(3):
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)

                # FedProx 正则项
                proximal_term = sum(p.norm().pow(2) for p in self.model.parameters())
                loss += mu / 2 * proximal_term

                loss.backward()
                optimizer.step()

                # 累计指标
                total_loss += loss.item() * y.size(0)
                pred = torch.argmax(outputs, dim=1)
                total_correct += (pred == y).sum().item()
                total_samples += y.size(0)

                # 打印每个 batch 的进度（可选）
                print(
                    f"Client {self.client_id}: Epoch {epoch + 1}/{3}, Batch {batch_idx + 1}/{len(self.trainloader)}, Loss: {loss.item():.4f}")

            # 打印每个 epoch 的训练结果
            epoch_loss = total_loss / total_samples
            epoch_accuracy = total_correct / total_samples
            print(
                f"Client {self.client_id}: Epoch {epoch + 1}/{3} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        return self.get_parameters(config), len(self.trainloader.dataset), {"loss": avg_loss, "accuracy": avg_accuracy}

    def evaluate(self, parameters, config):
        """本地评估"""
        self.set_parameters(parameters)
        self.model.eval()

        test_loss, correct, total = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.testloader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item() * y.size(0)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        test_loss /= total

        # 打印评估结果
        print(f"Client {self.client_id}: Evaluation completed. Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

        return float(test_loss), total, {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    client = FlowerClient(args.client_id)
    fl.client.start_client(
        server_address="192.168.1.196:8080",  # 替换为服务器实际IP
        client=client.to_client()
    )