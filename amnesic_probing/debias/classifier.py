import numpy as np
import torch
import wandb


# an abstract class for linear classifiers


class Classifier:
    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """

        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError


class SKlearnClassifier(Classifier):
    def __init__(self, m):
        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
            w = np.expand_dims(w, 0)

        return w


class PytorchClassifier(Classifier):
    def __init__(self, m: torch.nn.Module, device: str):
        self.m = m.to(device)
        self.device = device

    def eval(self, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        X_dev = torch.tensor(X_dev).float().to(self.device)
        Y_dev = torch.tensor(Y_dev).to(self.device)
        test_dataset = torch.utils.data.TensorDataset(X_dev, Y_dev)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096,
                                                 shuffle=False)
        acc = self._eval(testloader)
        # print("Eval accuracy: ", acc)
        return acc

    def _eval(self, testloader: torch.utils.data.DataLoader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                vectors, labels = data
                outputs = self.m(vectors)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on test: %d %%' % (
        #         100 * correct / total))
        return correct / total

    def get_probs(self, x: np.ndarray, y) -> np.ndarray:
        X = torch.tensor(x).float().to(self.device)
        Y = torch.tensor(y).to(self.device)
        test_dataset = torch.utils.data.TensorDataset(X, Y)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096,
                                                 shuffle=False)
        probs = self._get_probs(testloader)
        # print("Eval accuracy: ", acc)
        return probs

    def _get_probs(self, testloader: torch.utils.data.DataLoader):
        softmax_logits = []
        with torch.no_grad():
            for data in testloader:
                vectors, labels = data
                outputs = self.m(vectors)
                probs = torch.softmax(outputs.data, dim=1)
                softmax_logits.append(probs.cpu().numpy())

        return np.array(softmax_logits)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
              X_dev: np.ndarray, Y_dev: np.ndarray, epochs=1, save_path=None,
              use_wandb: bool = False) -> float:
        X_train = torch.tensor(X_train).to(self.device)
        Y_train = torch.tensor(Y_train).to(self.device)
        X_dev = torch.tensor(X_dev).to(self.device)
        Y_dev = torch.tensor(Y_dev).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                   shuffle=True)
        dev_dataset = torch.utils.data.TensorDataset(X_dev, Y_dev)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=2048,
                                                 shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.m.parameters(), lr=0.0001)

        acc = self._eval(dev_loader)
        best_acc = -1

        print("Dev accuracy before training: ", acc)
        if use_wandb:
            wandb.run.summary['dev_acc_no_ft'] = acc

        if save_path:
            torch.save(self.m, save_path)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.m(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            acc = self._eval(dev_loader)
            print("Dev acc during training:", acc)
            if use_wandb:
                wandb.log({'dev_acc': acc})
            if acc > best_acc:
                best_acc = acc
                if use_wandb:
                    wandb.run.summary['dev_best_acc'] = best_acc
                    wandb.run.summary['dev_best_epoch'] = epoch

                if save_path:
                    print("New best dev acc reached. Saving model to", save_path)
                    torch.save(self.m, save_path)

        print('Finished Training')

        acc = self._eval(dev_loader)

        print("Dev accuracy after training: ", acc)

        return acc

    # def get_weights(self, layer) -> Tuple[np.ndarray, np.ndarray]:
    #     if len(self.m) > 1:
    #         return self.m[1].weight.detach().cpu().numpy(), self.m[1].bias.detach().cpu().numpy()
