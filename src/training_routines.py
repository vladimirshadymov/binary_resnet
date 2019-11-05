import torch
import torch.nn.functional as F

def count_correct_preds(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

def train(args, model, device, train_loader, optimizer, epoch, penalty='cross_entropy'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if penalty == 'cross_entropy':
            loss = F.cross_entropy(output, target, reduction='mean')
        elif penalty == 'multi_margin':
            loss = F.multi_margin_loss(output, target, p=1, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, penalty='cross_entropy'):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if penalty == 'cross_entropy':
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
            elif penalty == 'multi_margin':
                test_loss += F.multi_margin_loss(output, target, p=1, reduction='sum').item() # sum up batch loss
            correct += count_correct_preds(output, target, topk=(1, 5))
            correct_top1 += correct[0]  
            correct_top5 += correct[1]

    test_loss /= len(test_loader.dataset)

    print('Average loss: {:.4f}, Accuracy top-1\%: {}/{} ({:.1f}%), Accuracy top-5\%: {}/{} ({:.1f}%)'.format(
        test_loss, correct_top1, len(test_loader.dataset),
        100. * correct_top1 / len(test_loader.dataset), correct_top5, len(test_loader.dataset),
        100. * correct_top5 / len(test_loader.dataset)))
