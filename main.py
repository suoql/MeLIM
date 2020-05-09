import os
import time
import argparse
import tensorflow as tf
from models.proposed import proposed_model


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='train',
                        help='actions: train, test')
    parser.add_argument('--model', type=str, default='proposed',
                        help='actions: metric, GAN_paired, GAN_unpaired, proposed')
    parser.add_argument('--suffix', type = str, default='p_upm', help='suffix for saving file names')
    parser.add_argument('--resuffix', type = str, default='p_upm', help='suffix reloaded model directory')
    parser.add_argument('--gpu', type = str, default='0', help='actions: choose gpu_id')    
    parser.add_argument('--img_size', type=int, default=32, help='size of images')
    parser.add_argument('--classes', type=str, default='CN_AD', help='binary classes: CN_AD, CN_MCI, AD_MCI or multiple classes: '' ')
    parser.add_argument('--portion', type=int, default=500, help = 'training paired data portion')
    parser.add_argument('--test_imp', type=int, default=1, help='testing set use imputed data or original test data, used to perform complete vs imputed exp')
   
    # training
    parser.add_argument('--max_step', type=int, default=300000, help='# of step for training')
    parser.add_argument('--save_interval', type=int, default=1000, help = '# of interval to save  model')
    parser.add_argument('--summary_interval', type=int, default=2000, help='number of step intervals to print')
    parser.add_argument('--learning_rate', type=int, default=2e-4, help = 'learning rate')
    parser.add_argument('--train_pair', type=str, default='adni_paired_train.h5', help = 'Training data')
    parser.add_argument('--train_unpair', type=str, default='adni_unpaired.h5', help = 'Training data')
    parser.add_argument('--valid_data', type=str, default='adni_paired_valid.h5', help = 'Validation data')
    parser.add_argument('--test_data', type=str, default='adni_paired_test.h5', help = 'Testing data')
    parser.add_argument('--trial', type=int, default=1, help = 'number of training trials, we repeat training for 5 times and report the mean')
    
    parser.add_argument('--isgen', type=int, default=0, help = 'use imputed data or original data (only complete samples) for metric learning')
    parser.add_argument('--gen_train', type=str, default='gen_train.h5', help = 'data with generated for metric learning')    
    parser.add_argument('--gen_test', type=str, default='gen_test.h5', help = 'data with generated for metric learning')

    parser.add_argument('--data_type', type=str, default='2D', help = '2D data or 3D data, we use 2D in our experiments.')
    parser.add_argument('--batch', type=int, default=100, help = 'batch size, should be even numbers for usage of metric learning')
    parser.add_argument('--channel', type=int, default=1, help = 'channel size')

    # Debug
    parser.add_argument('--save_image', type=int, default=1, help = 'Whether save images during training')
    parser.add_argument('--model_name', type=str, default='model', help = 'Model file name')
    parser.add_argument('--test_step', type=int, default=0, help = 'Test or predict model at this step')
    parser.add_argument('--random_seed', type=int, default=0, help = 'random seed')
    # network architecture
    parser.add_argument('--network_depth', type=int, default=5, help = 'network depth for U-Net')
    parser.add_argument('--class_num', type=int, default=1, help = 'output class number')
    parser.add_argument('--start_channel_num', type=int, default=16,
                         help = 'start number of outputs for the first conv layer')

    parser.add_argument('--conv_name', type=str, default='conv', help='Use which conv op in decoder: conv')
    parser.add_argument('--deconv_name', type=str, default='deconv', help = 'Use which deconv op in decoder: deconv')

    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    args.depth = args.img_size
    args.height = args.img_size
    args.width = args.img_size
    args.model_option = args.model    
    args.reload_step = args.test_step   #'Reload step to continue training'
    args.data_dir = '../Data/'+str(args.img_size)+'/'+args.data_type+'_'+args.classes+'/'   ##'Name of data directory'
    args.gen_dir = './GEN_data/'+str(args.classes)+'/' +str(args.img_size)+'_'+str(args.data_type)+'/'+args.resuffix+'_'+str(args.trial)#'directory for saving generated data'
    path = str(args.classes)+'/' + str(args.img_size)+'_'+str(args.data_type)+'/'+args.suffix +'_'+str(args.trial)
    args.logdir = './logdir/'+path          #'Log dir'
    args.modeldir = './modeldir/'+path      #'Model dir'
    args.sampledir = './samples/'+path      #'Sample directory'
    args.reloaddir = './modeldir/'+path     #'reload model directory


    print ('------------------------> trial %d <--------------------------' % args.trial)
    print ('dataset, portion, resuffix, isgen, test_imp:', args.classes, args.portion, args.resuffix, args.isgen, args.test_imp)
    if args.option not in ['train', 'test']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test")
    else:
        if args.model == 'proposed':
            print('start GAN paired_unpaired metric learning')
            model = proposed_model(sess, args)

    getattr(model, args.option)()


if __name__ == '__main__':
    tf.app.run()
