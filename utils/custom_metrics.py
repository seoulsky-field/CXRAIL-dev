import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


def accuracy(pred, y):
    correct = 0
    num_samples = y.shape[0]
    pred = np.where(pred < 0.5, 0., pred)
    pred = np.where(pred >= 0.5, 1., pred)  
    correct += (pred == y).sum().item()
    acc = 100 * correct / num_samples

    return acc


def get_accuracy_list(preds, labels):
    accuracy_list = []
    num_labels = labels.shape[1]
    for i in range(num_labels):
        acc = accuracy(preds[:,i],labels[:,i])
        accuracy_list.append(acc)
    
    return accuracy_list


def roc_curves(preds, labels, train_cols):
    fpr, tpr = dict(), dict()
    roc_auc_scores = dict()
    
    for name in train_cols:
        index = train_cols.index(name)
        for i in range(2):
            fpr[name, i], tpr[name, i], _ = roc_curve(labels[:,index]==i, preds[:,index])
            roc_auc_scores[name, i] = roc_auc_score(labels[:,index]==i, preds[:,index])

    return fpr, tpr, roc_auc_scores


def plot_roc_auc(fpr, tpr, roc_auc_scores, train_cols):
    plt.figure(figsize=(6,6))
    lw = 2
    class_name = ['negative', 'positive']
    train_cols = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
    colors = ["aqua", "darkorange", "cornflowerblue", "purple", "magenta"]

    for name in train_cols:
        color = colors.pop(0)
        plt.plot(fpr[name, 1], tpr[name, 1], color=color, lw=lw, label="{0} ({1}) (area = {2:0.2f})".format(name, class_name[1], roc_auc_scores[name, 1]))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("Receiver operating characteristic curve", fontsize=12)
    plt.tight_layout()    
    plt.show()


def cm_display(preds, labels, train_cols):
    preds = np.where(preds < 0.5, 0., preds)
    preds = np.where(preds >= 0.5, 1., preds) 
    num_labels = labels.shape[1]
    figs = dict()
    figs_norm = dict()

    for i in range(num_labels):
        cm = confusion_matrix(labels[:,i], preds[:,i])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        figs[i] = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
        figs_norm[i] = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['negative', 'positive'])
    
    
    for i in range(num_labels):
        figure = plt.figure(figsize=(3, 3))
        figs[i].plot()
        plt.title("Confusion Matrix - {0}".format(train_cols[i]), fontsize=12)
        figs_norm[i].plot()
        plt.title("Normalized Confusion Matrix - {0}".format(train_cols[i]), fontsize=12)
        plt.show()


def reports(preds, labels, train_cols):
    preds = np.where(preds < 0.5, 0., preds)
    preds = np.where(preds >= 0.5, 1., preds)  
    for i in range(len(train_cols)):
        label_name = train_cols[i]
        print("\n******************************************************")
        print(f'Classfication Reports {label_name}'.center(54))
        print("******************************************************")
        print(classification_report(labels[:,i], preds[:,i], target_names=['negative', 'positive']))
        print("******************************************************")


def custom_metrics(preds, labels, print_classification_result=True):
    train_cols = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']
    
    ### accuracy of each labels
    accuracy_list = get_accuracy_list(preds, labels)
    for i in range(len(train_cols)):
        print('---> [{0}] accuracy = {1}'.format(train_cols[i], accuracy_list[i]))
    print('===> Total accuracy = {0} <==='.format(np.mean(np.array(accuracy_list))))

    ### for roc curve
    fpr, tpr, roc_auc_scores = roc_curves(preds, labels, train_cols)
    # plot_roc_auc(fpr, tpr, roc_auc_scores, train_cols)

    ### confusion matrix
    # cm_display(preds, labels, train_cols)

    ### classification reports
    if print_classification_result:
        reports(preds, labels, train_cols)