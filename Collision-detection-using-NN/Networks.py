import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(6, 300)
        self.hidden_to_output = nn.Linear(300, 1)
        self.activation_function = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        x = self.input_to_hidden(input)
        x = self.relu(x)
        x = self.hidden_to_output(x)
        output = self.activation_function(x)
        return output


    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        model.eval()
        loss = 0
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                output = model(sample['input'])
                output = torch.reshape(output, (-1,))
                output_loss = loss_function(output, sample['label'])
                loss += output_loss.item()

        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
