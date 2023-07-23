import argparse
import pickle

import math
from tqdm import tqdm
import sys
import torch
from torch.nn.functional import interpolate as resize
import open3d as o3d

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    # fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


def gendata_from_pt(pt_path, out_path, benchmark='xview', part='eval', num_joint=524):
    ignored_samples = []
    sample_name = []
    sample_label = []
    adj_list_generated = False
    classes_file = open(pt_path + "../classes.txt")
    classes = classes_file.readlines()
    for i in range(0, len(classes), 1):
        classes[i] = classes[i].split('-')[0]

    for filename in os.listdir(pt_path):
        if filename in ignored_samples:
            continue
        action_class = generate_y(filename, classes)
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, 32, num_joint, 1), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = torch.permute(torch.load(os.path.join(pt_path, s)), (2, 0, 1)).view(3, 32, 524, 1)
        # data = resize(data, size=(32, 10475), mode='bilinear', align_corners=False).view(32, 10475, 3)
        # data = torch.FloatTensor(shrink_points_o3d(np.asarray(data), 500))
        if adj_list_generated is False:
            data_t = torch.permute(data, (3, 1, 2, 0)).squeeze()
            generate_adjacency_pair_inward(np.asarray(data_t)[0], path=out_path)
            adj_list_generated = True
        fp[i, :, 0:data.shape[1], :, :] = data

    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


def gendata_from_pt_skl(pt_path, out_path, benchmark='xview', part='eval'):
    ignored_samples = []
    sample_name = []
    sample_label = []
    adj_list_generated = False
    classes_file = open(pt_path + "../classes.txt")
    classes = classes_file.readlines()
    for i in range(0, len(classes), 1):
        classes[i] = classes[i].split('-')[0]

    for filename in os.listdir(pt_path):
        if filename in ignored_samples:
            continue
        action_class = generate_y(filename, classes)
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, 32, 25, 1), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = torch.permute(torch.load(os.path.join(pt_path, s)), (3, 1, 2, 0))
        if adj_list_generated is False:
            data_t = torch.permute(data, (3, 1, 2, 0)).squeeze()
            generate_adjacency_pair_inward(np.asarray(data_t)[0], path=out_path)
            adj_list_generated = True
        fp[i, :, 0:data.shape[1], :, :] = data

    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


def generate_y(filename, classes):

    for i in range(0, len(classes), 1):
        if classes[i] in filename:
            return int(i)
    return int(-1)

def shrink_points_o3d(original_sequence, target_num):
    result = []
    ratio = int(10475 / target_num)
    for i in range(0, len(original_sequence), 1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(original_sequence[i])
        pcd = pcd.uniform_down_sample(ratio)
        shape = len(pcd.points)

        if i == 0:
            result = np.asarray(pcd.points).reshape(1, shape, 3)
            # o3d.visualization.draw_geometries([pcd])
        else:
            result = np.append(result, np.asarray(pcd.points).reshape(1, -1, 3)).reshape(-1, shape, 3)
    return result.reshape(3, -1, shape, 1)


def generate_adjacency_pair_inward(frame, path):
    adj_list = []

    for i in range(0, len(frame), 1):
        dis_dict = {}
        for j in range(0, len(frame), 1):
            if i != j:
                dis_dict[j] = math.sqrt(
                    math.pow((frame[i][0] - frame[j][0]), 2) + math.pow((frame[i][1] - frame[j][1]), 2) + math.pow(
                        (frame[i][2] - frame[j][2]), 2))
        sorted_dis_dict = list(dict(sorted(dis_dict.items(), key=lambda item: item[1])).keys())

        if check_inward_adj_list(adj_list, i) is True:
            adj_list.append([i, sorted_dis_dict[0]])
            adj_list.append([i, sorted_dis_dict[1]])
        # If there is more, keep adding, else stop
        k = 2
        while dis_dict[sorted_dis_dict[k]] < 1.5 * dis_dict[sorted_dis_dict[0]]:
            if check_inward_adj_list(adj_list, i) is True:
                adj_list.append([i, sorted_dis_dict[k]])
            k += 1
    torch.save(adj_list, f'{path}/inward.pt')


def check_inward_adj_list(l, node):
    for pair in l:
        if pair[1] == node:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/mnt/h/Datasets/NTU/skl_pt_11_classes/all/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/ntu_less_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/mnt/h/Datasets/NTU/aagcn_skl/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    # d = np.load('../data/ntu_test/xsub/train_data_joint.npy')

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            # gendata(
            #     data_path=arg.data_path,
            #     out_path=out_path,
            #     benchmark=b,
            #     ignored_sample_path=arg.ignored_sample_path,
            #     part=p)
            gendata_from_pt_skl(
                pt_path=arg.data_path,
                out_path=out_path,
                benchmark=b,
                part=p)
