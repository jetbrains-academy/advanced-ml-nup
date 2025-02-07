import pandas as pd

from Unsupervised.EMforDSPoints30.task import DawidSkeneEM


def majority_voting(labels: pd.DataFrame):
    """
    Aggregate labels using majority voting.
    :param labels: DataFrame with columns 'object', 'user', and 'label'
    :return: DataFrame with columns 'object' and 'label'
    """
    majority_vote = None
    vote_counts = pd.crosstab(index=labels['object'], columns=labels['label'])
    majority_vote = vote_counts.idxmax(axis=1)
    return majority_vote


def get_accuracy(labels_pred: pd.DataFrame, labels_true: pd.DataFrame) -> float:
    labels_dict = labels_true.groupby('object')['label'].apply(set)
    labels_pred = labels_pred.rename('label').reset_index()
    labels_pred = labels_pred[labels_pred.object.isin(labels_dict.index)]
    matches = labels_pred.apply(lambda p: p.label in labels_dict.loc[p.object], axis=1)
    return sum(matches) / len(labels_pred)


if __name__ == '__main__':
    toloka2_crowd = pd.read_csv('tlk2_crowd_labels.tsv',
                                names=['user', 'object', 'label'], sep='\t')
    toloka2_golden = pd.read_csv('tlk2_golden_labels.tsv',
                                 names=['object', 'label'], sep='\t')
    toloka2_golden.set_index('object', inplace=True)

    toloka2_mv = majority_voting(toloka2_crowd)
    print(f"Majority voting acc = {get_accuracy(toloka2_mv, toloka2_golden):.2f}")

    model = DawidSkeneEM(toloka2_crowd, 15)
    model.fit()
    toloka2_em = model.predict()
    print(f"EM acc = {get_accuracy(toloka2_em, toloka2_golden):.2f}")
