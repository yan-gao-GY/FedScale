import pickle


dataset = 'test'
with open('/datasets/FedScale/leaf_femnist/client_data_mapping/{}.csv'.format(dataset), 'w') as fw:
    header = 'client_id,sample_path,index,label_id' + '\n'
    fw.write(header)
    for cid in range(178):
        f_path = '/datasets/FedScale/leaf_femnist/data/{}/{}.pickle'.format(cid, dataset)
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            y = data['y']
            data_len = len(y)

            for ind in range(data_len):
                line = '{},{},{},{}'.format(cid, f_path, ind, y[ind])
                fw.write(line + '\n')





import os
import pickle

len_list = []
for f_name in os.listdir('./data'):
    with open('./data/{}/train.pickle'.format(f_name), 'rb') as f:
        d = pickle.load(f)
        len_list.append(len(d['x']))