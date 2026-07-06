import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from distillation.utils import PseudoDataset
from distillation.dynamicDistiller import DynamicTemperatureDistiller


teacher = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
student = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10))


dataset = PseudoDataset(size=(32,))
dataloader = DataLoader(dataset, batch_size=4)


optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
distill_criterion = nn.KLDivLoss(reduction='batchmean')



distiller = DynamicTemperatureDistiller(alpha=0.5)



metrics = distiller.train_step(student, teacher, dataloader, optimizer, criterion, distill_criterion)
print("result = ", metrics)
