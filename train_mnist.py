 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from distillation.dynamicDistiller import DynamicTemperatureDistiller

#MINST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)


teacher = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*7*7, 128), nn.ReLU(),
    nn.Linear(128, 10)
)


student = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8*14*14, 32), nn.ReLU(),
    nn.Linear(32, 10)
)

def train_teacher(model, loader, epochs=3):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Teacher Epoch {ep+1}: Loss = {total_loss/len(loader):.4f}")

print("train MNIST on teacher ")
train_teacher(teacher, train_loader, epochs=3)


optimizer = optim.Adam(student.parameters(), lr=0.001)
criterion_hard = nn.CrossEntropyLoss()
criterion_soft = nn.KLDivLoss(reduction='batchmean')

distiller = DynamicTemperatureDistiller(
    alpha=0.5,
    t_base=2.0,
    beta=1.5
)

for epoch in range(5):
    metrics = distiller.train_step(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        optimizer=optimizer,
        objective=criterion_hard,
        distillObjective=criterion_soft
    )
    print(f"Epoch {epoch+1}: Loss = {metrics['Train/Loss']:.4f}, Acc = {metrics['Train/Accuracy']:.4f}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

acc_test = evaluate(student, test_loader)
print(f"\n accuracy on student {acc_test*100:.2f}%")
