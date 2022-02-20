import matplotlib.pyplot as plt


def show_acc_loss(numerical_settings: dict, train_acc: list, train_loss: list, file_name: str, labels=None):
    relation_num, children_num = numerical_settings['relation_num'], numerical_settings['father_num']
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if isinstance(train_loss[0], list):
        loss_labels = ['Train loss', 'Valid loss']
        for loss, label in zip(train_loss, loss_labels):
            epoch_num = list(range(1, len(loss) + 1))
            plt.plot(epoch_num, loss, label=label)
    else:
        epoch_num = list(range(1, len(train_loss) + 1))
        plt.plot(epoch_num, train_loss, label='Train Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title(f'#fathers={children_num}: Rel. memorization Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if isinstance(train_acc[0], list):
        if labels:
            label = labels
        else:
            label = ['task1', 'task2', 'task3']
        for acc, label in zip(train_acc, label):
            epoch_num = list(range(1, len(acc) + 1))
            plt.plot(epoch_num, acc, label=label)
    else:
        epoch_num = list(range(1, len(train_acc) + 1))
        plt.plot(epoch_num, train_acc, label='Train Acc')
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.title(f'#fathers={children_num}: Rel. memorization Acc.')
    plt.legend()
    # plt.grid(linestyle='dotted')

    plt.tight_layout()
    plt.savefig(file_name)
    print('Recording Acc & Loss done')
