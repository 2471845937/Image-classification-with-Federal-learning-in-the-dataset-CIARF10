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
        self.accuracies = []  # 测试准确率
        self.train_accuracies = []  # 训练准确率（新增）
        self.train_losses = []  # 训练损失（新增）

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

        client_sockets = []
        for _ in range(num_clients):
            client_socket, addr = server_socket.accept()
            print(f"Connected to {addr}")
            client_sockets.append(client_socket)

        for round_num in range(num_rounds):
            self.global_round = round_num + 1
            client_updates = []
            train_acc_list = []
            train_loss_list = []

            # 发送训练指令和全局模型
            for sock in client_sockets:
                try:
                    sock.sendall(b'TRAIN')
                    model_data = pickle.dumps(self.model.state_dict())
                    sock.sendall(len(model_data).to_bytes(4, 'big'))
                    sock.sendall(model_data)
                except ConnectionResetError:
                    print("Client connection lost!")
                    continue

            # 接收客户端更新
            for sock in client_sockets:
                try:
                    data_len = int.from_bytes(sock.recv(4), 'big')
                    if not data_len:
                        continue

                    data = b''
                    while len(data) < data_len:
                        packet = sock.recv(data_len - len(data))
                        if not packet:
                            break
                        data += packet

                    if len(data) == data_len:
                        client_data = pickle.loads(data)
                        client_updates.append(client_data['params'])
                        train_acc_list.append(client_data['train_acc'])
                        train_loss_list.append(client_data['train_loss'])
                    else:
                        print("Received incomplete data!")
                except ConnectionResetError:
                    print("Failed to receive client update!")
                    continue

            # 聚合参数
            if client_updates:
                self.aggregate(client_updates)

            # 计算平均训练指标
            avg_train_acc = sum(train_acc_list) / len(train_acc_list) if train_acc_list else 0
            avg_train_loss = sum(train_loss_list) / len(train_loss_list) if train_loss_list else 0
            self.train_accuracies.append(avg_train_acc)
            self.train_losses.append(avg_train_loss)

            # 测试全局模型
            test_loss, test_acc = test(self.model, self.test_loader, self.device)
            self.accuracies.append(test_acc)

            # 打印本轮结果
            print(f"\nRound {self.global_round} Results:")
            print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {avg_train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

        # 发送终止指令
        for sock in client_sockets:
            try:
                sock.sendall(b'EXIT')
                sock.close()
            except:
                pass

        # 绘制三个图表
        rounds = range(1, num_rounds + 1)

        # 测试准确率
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.accuracies, marker='o')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        plt.grid(True)
        plt.savefig('fl_test_accuracy.png')
        plt.close()

        # 训练准确率
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.train_accuracies, marker='o', color='orange')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Train Accuracy')
        plt.grid(True)
        plt.savefig('fl_train_accuracy.png')
        plt.close()

        # 训练损失
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.train_losses, marker='o', color='red')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.grid(True)
        plt.savefig('fl_train_loss.png')
        plt.close()

        server_socket.close()
        print("Training completed!")


if __name__ == "__main__":
    server = FederatedServer()
    server.run(num_rounds=30)
