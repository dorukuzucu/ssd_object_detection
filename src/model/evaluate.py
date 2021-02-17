import torch


def eval(model, dataloader, criteria, gpu_f=True):
    """
    :param model: model to be tested
    :param dataloader: dataloader for testing set
    :param criteria: loss function to be used
    :return: return predictions and targets as tuple
    """
    eval_loss = float(0.0)
    model.eval()

    locations = []
    confidences = []

    with torch.no_grad():
        for data, label in dataloader:
            if gpu_f:
                data = data.to(torch.device('cuda:0'))
                label = label.to(torch.device('cuda:0'))

            out = model(data)
            loss = criteria.calculate(out, label)

            locs, confs = out
            locations.append(locs)
            confidences.append(confs)

            eval_loss += float(loss.item())
        print("Validation: Loss:{} ".format(eval_loss / len(dataloader)))

        return (eval_loss / len(dataloader))
