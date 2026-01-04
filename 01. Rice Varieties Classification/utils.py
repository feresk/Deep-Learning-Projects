import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize 
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn import metrics


N_CLASSES = 5
CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
CLASS_TO_NUM = {c:i for c,i in zip(CLASSES, range(N_CLASSES))}
NUM_TO_CLASS = {i:c for c,i in zip(CLASSES, range(N_CLASSES))}

def plot_pr_curve(ax, y_score, y_test, name='Micro-avg', type='mean', n=N_CLASSES, map_dict=NUM_TO_CLASS, alpha=.5, chance_level=True):
    """"""
    y_test_bin = label_binarize(y_test, classes=list(range(n)))
    precision, recall, average_precision = dict(), dict(), dict()
    for i in range(n):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_test_bin[:,i], y_score[:,i])
        average_precision[i] = metrics.average_precision_score(y_test_bin[:,i], y_score[:,i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test_bin, y_score, average="micro")
    if type in ['per_class', 'both']:
        for i in range(n):
            pr_class = metrics.PrecisionRecallDisplay( recall=recall[i], precision=precision[i], 
                                            average_precision=average_precision[i],
                                            estimator_name=map_dict[i].capitalize() )
            pr_class.plot(ax=ax, alpha=.5)
    if type in ['mean', 'both']:
        pr_mean = metrics.PrecisionRecallDisplay( recall=recall["micro"], precision=precision["micro"], 
                                            average_precision=average_precision["micro"], 
                                            prevalence_pos_label=y_test_bin.sum()/y_test_bin.size, 
                                            estimator_name=name )
        pr_mean.plot(ax=ax, plot_chance_level=chance_level, despine=True)
    ax.legend(fontsize=10, ncol=1 if type=='mean' else 2)

def plot_roc_curve(ax, y_score, y_test, name="Micro-average OvR", type='mean', n=N_CLASSES, map_dict=NUM_TO_CLASS, alpha=.5):
    """"""
    y_test_bin = label_binarize(y_test, classes=list(range(n)))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:,i], y_score[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
    if type in ['per_class', 'both']:
        for i in range(n):
            roc_class = metrics.RocCurveDisplay(fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i], estimator_name=map_dict[i].capitalize())
            roc_class.plot(ax=ax, alpha=.5)
    if type in ['mean', 'both']:
        roc_mean = metrics.RocCurveDisplay(fpr=fpr['micro'], tpr=tpr['micro'], roc_auc=roc_auc['micro'], estimator_name=name)
        roc_mean.plot(ax=ax)
    ax.legend(fontsize=10, ncol=1 if type=='mean' else 2)
    
def plot_calibration_curve(ax, y_score, y_test, name="Micro-average OvR", bins=5, type='mean', n=N_CLASSES, map_dict=NUM_TO_CLASS, alpha=.5):
    """"""
    y_test_bin = label_binarize(y_test, classes=list(range(n)))
    prob_true, prob_pred = dict(), dict()
    for i in range(n):
        prob_true[i], prob_pred[i] = calibration_curve(y_test_bin[:,i], y_score[:,i], n_bins=bins-2 if bins>2 else bins)
    prob_true['micro'], prob_pred['micro'] = calibration_curve(y_test_bin.ravel(), y_score.ravel(), n_bins=bins)
    if type in ['per_class', 'both']:
        for i in range(n):
            calib_class = CalibrationDisplay(prob_true=prob_true[i], prob_pred=prob_pred[i], y_prob=y_score[i],
                                             estimator_name=map_dict[i].capitalize())
            calib_class.plot(ax=ax, alpha=.5)
    if type in ['mean', 'both']:
        calib_mean = CalibrationDisplay(prob_true=prob_true['micro'], prob_pred=prob_pred['micro'], y_prob=y_score.ravel(),
                                        estimator_name=name)
        calib_mean.plot(ax=ax)
    ax.legend(fontsize=10, ncol=1)
    
def eval_summary(y_score, y_proba, y_test, size=[14,14], names=['Micro-avg','Micro-average OvR', 'Micro-average OvR'], 
                 bins=5, types=['both', 'both', 'both'], title='', n=N_CLASSES, map_dict=NUM_TO_CLASS, alpha=.5):
    """"""
    fig, ax = plt.subplot_mosaic([['pr', 'roc'],['cal', 'cal']], figsize=size, layout='constrained')
    plot_pr_curve(ax['pr'], y_score, y_test, name=names[0], type=types[0], n=n, map_dict=map_dict, alpha=alpha)
    plot_roc_curve(ax['roc'], y_score, y_test, name=names[1], type=types[1], n=n, map_dict=map_dict, alpha=alpha)
    plot_calibration_curve(ax['cal'], y_proba, y_test, name=names[2], type=types[2], bins=bins, n=n, map_dict=map_dict, alpha=alpha)
    fig.suptitle(title, size=18)
        

