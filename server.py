import socket
import pickle
import torch
import matplotlib.pyplot as plt
from model import create_model, test
from data_loader import get_datasets


class FederatedServer:
    def __init__(self):
        self.model = create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.global_round = 0
        self.accuracies = []
        self.train_losses = []
        self.train_accuracies = []
        self.best_accuracy = 0.0

        # Get test loader
        _, test_data = get_datasets(0)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    def aggregate(self, client_params):
        global_dict = self.model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.stack([param[key].float() for param in client_params]).mean(0)
        self.model.load_state_dict(global_dict)

    def run(self, num_rounds=10, num_clients=2):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('192.168.1.203', 12345))
        server_socket.listen(num_clients)

        print(f"Server listening on port 12345...")

        # 建立所有客户端的长连接
        client_sockets = []
        for _ in range(num_clients):
            client_socket, addr = server_socket.accept()
            print(f"Connected to {addr}")
            client_sockets.append(client_socket)

        for round_num in range(num_rounds):
            self.global_round = round_num + 1
            client_updates = []

            # 向所有客户端发送训练指令和全局模型
            for sock in client_sockets:
                try:
                    # 发送训练指令
                    sock.sendall(b'TRAIN')

                    # 发送全局模型
                    model_data = pickle.dumps(self.model.state_dict())
                    sock.sendall(len(model_data).to_bytes(4, 'big'))
                    sock.sendall(model_data)
                except ConnectionResetError:
                    print("Client connection lost!")
                    continue

            # 接收所有客户端的参数更新
            for sock in client_sockets:
                try:
                    # 接收参数数据长度
                    data_len = int.from_bytes(sock.recv(4), 'big')
                    if not data_len:
                        continue

                    # 接收完整数据
                    data = b''
                    while len(data) < data_len:
                        packet = sock.recv(data_len - len(data))
                        if not packet:
                            break
                        data += packet

                    if len(data) == data_len:
                        client_updates.append(pickle.loads(data))
                    else:
                        print("Received incomplete data!")
                except ConnectionResetError:
                    print("Failed to receive client update!")
                    continue

            # 聚合参数并收集训练指标
            if client_updates:
                client_params = [update['params'] for update in client_updates]
                self.aggregate(client_params)

                # 计算平均训练指标
                avg_train_loss = sum(update['loss'] for update in client_updates) / len(client_updates)
                avg_train_acc = sum(update['accuracy'] for update in client_updates) / len(client_updates)
                self.train_losses.append(avg_train_loss)
                self.train_accuracies.append(avg_train_acc)

            # 测试全局模型
            test_loss, test_acc = test(self.model, self.test_loader, self.device)
            self.accuracies.append(test_acc)

            # 保存最佳模型
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with accuracy {self.best_accuracy:.2f}%")

            print(f"\nRound {self.global_round} Results:")
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

        # 发送终止指令
        for sock in client_sockets:
            try:
                sock.sendall(b'EXIT')
                sock.close()
            except:
                pass

        # 绘制准确率曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_rounds + 1), self.accuracies, marker='o', label='Test')
        plt.plot(range(1, num_rounds + 1), self.train_accuracies, marker='s', label='Train')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_rounds + 1), self.train_losses, marker='o', color='orange')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('fl_performance.png')
        plt.close()

        server_socket.close()
        print("Training completed!")


if __name__ == "__main__":
    server = FederatedServer()
    server.run(num_rounds=30)

if __name__ == "__main__":
    server = FederatedServer()
    server.run(num_rounds=30)
