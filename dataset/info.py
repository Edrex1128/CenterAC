
class DatasetInfo(object):
    dataset_info = {
        'coco_train': {
            'name': 'coco',
            'image_dir': 'COCO_dataset/train2017',
            'anno_dir': 'coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'coco_val': {
            'name': 'coco',
            'image_dir': 'COCO_dataset/val2017',
            'anno_dir': 'coco/annotations/instances_val2017.json',
            'split': 'val'
        },
        'coco_test': {
            'name': 'coco',
            'image_dir': 'COCO_dataset/test2017',
            'anno_dir': 'coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'sbd_train': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'sbd_val': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'kitti_train': {
            'name': 'kitti',
            'image_dir': 'data/kitti/training/image_2', 
            'anno_dir': 'data/kitti/training/instances_train.json', 
            'split': 'train'
        }, 
        'kitti_val': {
            'name': 'kitti',
            'image_dir': 'data/kitti/testing/image_2', 
            'anno_dir': 'data/kitti/testing/instances_val.json', 
            'split': 'val'
        },
        'cityscapes_train': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'cityscapes_val': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'cityscapesCoco_val': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/cityscapes/leftImg8bit/val',
            'anno_dir': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'cityscapes_test': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit/test', 
            'anno_dir': 'data/cityscapes/annotations/test', 
            'split': 'test'
        },
        'mydata_train': {
            'name': 'mydata',
            'image_dir': 'Mydataset/JPEGImages/train2017',
            'anno_dir': 'Mydataset/annotations/train.json',
            'split': 'train'
        },
        'mydata_val': {
            'name': 'mydata',
            'image_dir': 'Mydataset/JPEGImages/val2017',
            'anno_dir': 'Mydataset/annotations/val.json',
            'split': 'val'
        },
        'mydata_test': {
            'name': 'mydata',
            'image_dir': 'Mydataset/JPEGImages/val2017',
            'anno_dir': 'Mydataset/annotations/val.json',
            'split': 'test'
        }
    }
