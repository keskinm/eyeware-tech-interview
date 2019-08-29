import torch
from torch.utils.data import DataLoader
from models.seven_conv import seven_conv
from models.two_conv import two_conv
from numpy import linalg as LA
from dataset.simulated_dataset import SimulatedDataset


set = SimulatedDataset(work_dir='./data/task3/images', len_data=20)
set_loader = DataLoader(set, batch_size=4,
                        shuffle=True, num_workers=0)


seven_conv_model = seven_conv()
two_conv_model = two_conv()


model = two_conv_model
model = model.double()


# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
#                                    model.parameters()),
#                             lr=0.005,
#                             momentum=0.9)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=0.005)

criterion = torch.nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
          model,
          set_loader,
          criterion,
          optimizer,
          device):

    losses=[]

    epoch_n = 50
    model = model.train()

    for epoch in range(1, epoch_n + 1):
        for batch_id, (image, label) in enumerate(set_loader):
            label, image = label.to(device), image.to(device)
            output = model(image)
            loss = criterion(output, label.double())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_accuracy(model, data_set, device):
    model = model.eval()
    with torch.no_grad():
        mse = 0
        total = 0
        for batch_id, (image, label) in enumerate(data_set):
            image = image.to(device)
            label = label.to(device)
            outputs = model(image).to(device)
            se = LA.norm(outputs-label.double())
            print(se)
            total += label.shape[0]/4*2
            mse += se
    print(mse)
    return mse

train(model=model, set_loader=set_loader, criterion=criterion, optimizer=optimizer, device=device)
compute_accuracy(model, set_loader, device)