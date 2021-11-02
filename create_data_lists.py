from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['../../../MWDatabase/SR/SRGAN/COCO2014/train2014',
                                     '../../../MWDatabase/SR/SRGAN/COCO2014/val2014'],
                      test_folders=['../../../MWDatabase/SR/SRGAN/BSD100',
                                    '../../../MWDatabase/SR/SRGAN/Set5',
                                    '../../../MWDatabase/SR/SRGAN/Set14'],
                      min_size=100,
                      output_folder='./data/')
