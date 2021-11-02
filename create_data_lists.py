from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['E:/source/MWDatabase/SR/SRGAN/COCO2014/train2014',
                                     'E:/source/MWDatabase/SR/SRGAN/COCO2014/val2014'],
                      test_folders=['E:/source/MWDatabase/SR/SRGAN/BSD100',
                                    'E:/source/MWDatabase/SR/SRGAN/Set5',
                                    'E:/source/MWDatabase/SR/SRGAN/Set14'],
                      min_size=100,
                      output_folder='./data/')
