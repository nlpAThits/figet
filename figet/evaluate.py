import sys
import torch


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def strict(true_and_prediction):
    """
    Correct: all types must be predicted exactly equal to the label
    """
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if true_labels.size() != predicted_labels.size():
            continue
        correct_num += torch.all(true_labels == predicted_labels).item()
    precision = recall = correct_num / num_entities
    return precision, recall, f1(precision, recall)


def loose_macro(true_and_prediction):
    """Metrics at mention level.
    Takes an average of the metrics on the amount of mentions"""
    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        numerator = len(set([i.item() for i in predicted_labels]).intersection(set([j.item() for j in true_labels])))
        p += numerator / float(len(predicted_labels))
        r += numerator / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1(precision, recall)


def loose_micro(true_and_prediction):
    """Metrics at type/class level.
    Correct types of all types on all mentions"""
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set([i.item() for i in predicted_labels]).intersection(set([j.item() for j in true_labels])))
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)


def evaluate(true_and_prediction, verbose=False):
    ret = ""
    p, r, f = strict(true_and_prediction)
    if verbose:
        ret += "| strict (%.2f, %.2f, %.2f) " %(p*100, r*100, f*100)
    else:
        ret += "%.2f\t%.2f\t%.2f\t" % (p * 100, r * 100, f * 100)
    p, r, f = loose_macro(true_and_prediction)
    if verbose:
        ret += "| macro (%.2f, %.2f, %.2f) " %(p*100, r*100, f*100)
    else:
        ret += "%.2f\t%.2f\t%.2f\t" % (p * 100, r * 100, f * 100)
    p, r, f = loose_micro(true_and_prediction)
    if verbose:
        ret += "| micro (%.2f, %.2f, %.2f) |" %(p*100, r*100, f*100)
    else:
        ret += "%.2f\t%.2f\t%.2f\t" % (p * 100, r * 100, f * 100)
    return ret


def raw_evaluate(true_and_prediction):
    metrics = [strict, loose_macro, loose_micro]
    res = []
    for metric in metrics:
        p, r, f = metric(true_and_prediction)
        res.append((p * 100, r * 100, f * 100))
    return res


if __name__ == "__main__":
    file = open(sys.argv[1])
    true_and_prediction = []
    for line in file:
        temp = line.split("\t")
        if len(temp) == 1:
            true_labels = temp[0].split()
            predicted_labels = []
        else:
            true_labels, predicted_labels = temp
            true_labels = true_labels.split()
            predicted_labels = predicted_labels.split()
            true_and_prediction.append((true_labels,predicted_labels))
    #for each in true_and_prediction:
        #print(each)
    print("     strict (p,r,f1): " + strict(true_and_prediction))
    print("loose macro (p,r,f1): " + loose_macro(true_and_prediction))
    print("loose micro (p,r,f1): " + loose_micro(true_and_prediction))
    file.close()
