from sklearn.svm import LinearSVC
import scipy.stats

from preprocess import *
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=Path)
parser.add_argument('train_data_path', type=Path)
parser.add_argument('val_data_path', type=Path)

if __name__ == '__main__':
    args = parser.parse_args()

    a = load_config(args.train_dir).setdefaults(
        train_dir=args.train_dir,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        **train_defaults
    )
    assert a.use_scikit_learn and a.loss == 'pairwise_svm' and a.device == 'cpu'

    print('Loading data')
    train_data = np.load(a.train_data_path)
    train_data = {k: train_data[k] for k in ['linear_static_features', 'linear_dynamic_features', 'costs', 'agents']}
    val_data = dict(np.load(a.val_data_path))
    np.random.seed(0)

    features = train_data['linear_static_features'], train_data['linear_dynamic_features']
    features_max = np.concatenate([x.max(axis=(0, 1)) for x in features], dtype=np.float32)
    val_data['features_min'] = features_min = np.concatenate([x.min(axis=(0, 1)) for x in features], dtype=np.float32)
    val_data['features_range'] = features_range = np.clip(features_max - features_min, 1e-6, None)

    net = eval(a.network + 'Network')(a, val_data)

    state_dict = net.state_dict()
    torch.save(dict(step=0, time=0, net=state_dict), (a.train_dir / 'models').mk() / '0.pth')

    print('Preprocessing linear dataset')
    linear_features, costs = preprocess_linear_dataset(train_data, a, features_min, features_range)

    labels = np.clip(costs[:, :, 0] - costs[:, :, 2], -np.inf, 0 if a.mse_clip else np.inf) # (NL, S)
    if a.label_discretization:
        percentiles = np.percentile(labels, a.label_discretization, axis=-1, keepdims=True) # (num_percentiles, NL, 1)
        # new discretized labels, lower is better
        labels = (labels > percentiles).sum(axis=0) # (NL, S)

    NL, S, num_features = linear_features.shape

    diff_features = linear_features.reshape(NL, S, 1, num_features) - linear_features.reshape(NL, 1, S, num_features)
    diff_labels = np.sign(labels.reshape(NL, S, 1) - labels.reshape(NL, 1, S))

    mask = diff_labels != 0
    diff_features = diff_features[mask]
    diff_labels = diff_labels[mask]
    diff_labels[diff_labels == -1] = 0 # Relabel -1 as 0

    model = LinearSVC(random_state=0, dual=False, max_iter=a.train_steps)

    print('Fitting model')
    fit_start_time = time()
    model.fit(diff_features, diff_labels)
    fit_time = time() - fit_start_time

    state_dict['bias'].fill_(model.intercept_[0])
    state_dict['out_fc.bias'].fill_(0)
    state_dict['out_fc.weight'][:] = torch.tensor(model.coef_)

    checkpoint = dict(step=a.train_steps, time=fit_time, net=state_dict)
    torch.save(checkpoint, (a.train_dir / 'models').mk() / f'{a.train_steps}.pth')

    preds = diff_features.dot(model.coef_[0]) + model.intercept_

    for fn in [scipy.stats.pearsonr, scipy.stats.spearmanr, scipy.stats.kendalltau]:
        print(fn(preds, diff_labels))
