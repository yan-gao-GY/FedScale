import pickle


dataset = 'test'
with open('/datasets/FedScale/leaf_shakespeare/client_data_mapping/{}.csv'.format(dataset), 'w') as fw:
    header = 'client_id,sample_path,index,label' + '\n'
    fw.write(header)
    for cid in range(27):
        f_path = '/datasets/FedScale/leaf_shakespeare/data/{}/{}.pickle'.format(cid, dataset)
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            y = data['y']
            data_len = len(y)

            for ind in range(data_len):
                line = '{},{},{},{}'.format(cid, f_path, ind, y[ind])
                fw.write(line + '\n')
