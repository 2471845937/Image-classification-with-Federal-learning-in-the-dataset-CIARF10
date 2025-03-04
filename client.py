import socket
import pickle
import argparse
import torch
from torch import nn, optim
from data_loader import get_dataloaders
from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.train_loader, self.test_loader = get_dataloaders(client_id)
        self.model = create_model().to(device)

    def local_train(self, global_params, local_epochs=3):
        self.model.load_state_dict(global_params)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(local_epochs):
            total_loss = 0
            correct = 0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            train_loss = total_loss / len(self.train_loader)
            train_acc = 100. * correct / len(self.train_loader.dataset)
            print(f"Client {self.client_id} - Epoch {epoch + 1}: "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        return self.model.state_dict()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('192.168.1.203', 12345))

        while True:
            # 等待服务器指令
            try:
                command = sock.recv(5)
                if not command:
                    continue

                command = command.decode()
                if command == 'EXIT':
                    break
                elif command != 'TRAIN':
                    continue

                # 接收全局模型
                data_len = int.from_bytes(sock.recv(4), 'big')
                data = b''
                while len(data) < data_len:
                    packet = sock.recv(data_len - len(data))
                    if not packet:
                        break
                    data += packet

                if len(data) != data_len:
                    print("Received incomplete model parameters!")
                    continue

                global_params = pickle.loads(data)

                # 本地训练
                updated_params = self.local_train(global_params)

                # 发送更新参数
                data = pickle.dumps(updated_params)
                sock.sendall(len(data).to_bytes(4, 'big'))
                sock.sendall(data)

            except ConnectionResetError:
                print("Connection to server lost!")
                break
            except KeyboardInterrupt:
                break

        sock.close()
        print(f"Client {self.client_id} exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    args = parser.parse_args()

    client = FederatedClient(args.client_id)
    client.run()
