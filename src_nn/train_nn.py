import torch


def train(model, criterion, optimizer, data_loader, device, mode='1to1'):
    softmax = torch.nn.Softmax(dim=-1)
    model.train()
    running_loss = .0
    for batch in data_loader:
        optimizer.zero_grad()
        pred = model(batch, device)
        if mode == '1to1':
            loss = criterion(pred, torch.argmax(batch['answer'], dim=-1).to(device))
        elif mode == '1toN':
            pred = softmax(pred)
            loss = criterion(pred, batch['answer'].float().to(device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)
