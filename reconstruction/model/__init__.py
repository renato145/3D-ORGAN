import click, os, multiprocessing, torch, pickle
import numpy as np
from collections import deque
from time import time
from keras.utils import plot_model
from .custom_loader import CustomLoader
from .models import make_generator, make_discriminator
from .gan_utils import build_gan_arch
from ..utils import format_time

OPTS = ['voxels-ugan', 'voxels-u',
        'voxels-vgan', 'voxels-v',
        'voxels-usegan', 'voxels-use']

def check_epoch(file):
    try:
        with open(file) as f:
            f.seek(0, 2)
            f.seek(f.tell() - 128)
            epoch = int(f.readlines()[-1].split(', ')[0])
    except:
        epoch = 0
    
    return epoch

def get_batch(iter_loader):
    x = next(iter_loader)
    return x[0].numpy(), x[1].numpy(), x[2].numpy()

def write_generator_loss(file, epoch, v, gan=True):
    lines = ''
    if not os.path.exists(file):
        if gan:
            lines += 'epoch,generator_loss,gan_loss,l1_loss\n'
        else:
            lines += 'epoch,l1_loss\n'
    lines += f'{epoch}, '
    lines += ', '.join('%f' % i for i in v) if gan else f'{v}'
    lines += '\n'
    with open(file, 'a') as f:
        f.writelines(lines)

def write_discriminator_loss(file, epoch, v):
    lines = ''
    if not os.path.exists(file):
        lines += 'epoch,wgp_loss,disc_real,disc_fake,gp\n'
    lines += f'{epoch}, '
    lines += ', '.join('%f' % i for i in v)
    lines += '\n'
    with open(file, 'a') as f:
        f.writelines(lines)
        
def write_training_log(file, epoch, v):
    lines = ''
    if not os.path.exists(file):
        lines += 'epoch,l1_loss_train,l1_loss_test\n'
    lines += f'{epoch}, '
    lines += ', '.join('%f' % i for i in v)
    lines += '\n'
    with open(file, 'a') as f:
        f.writelines(lines)


class LoadModel(object):
    def __init__(self, data_file, out_path, batch_size=64, num_workers=-1,
                 training_ratio=5, gradient_penalty=10, loss='l1', loss_multiply=100,
                 overwrite=False, config_file='config.txt', opt='voxels-ugan',
                 evaluate_mode=False):
        '''
        inputs:
            data_file: path of .npy file or name defined in config.txt
                       Can be None.
            out_path: directory to save/load model files
            batch_size: int
            num_workers: defaults to -1 (all processors)
            training_ratio: x discriminator iters per each generator iter
            gradient_penalty: float
            overwrite: bool, remove existing training files if they exist
            config_file: path for a configuration file
            opts: str, model arquitecture:
                - 'voxels-v' (vanilla)
                - 'voxels-vgan'
                - 'voxels-u' (u type encode)
                - 'voxels-ugan'
                - 'voxels-use' (u + squeeze&excite)
                - 'voxels-usegan'
            evaluate_mode: bool, if True will only load the generator.
            
        --------------------------------------    
        configuration file ex:
            modelnet10: /path/to/file.npy
        
        '''

        assert opt in OPTS, f'The opt {opt!r} not in ({", ".join(OPTS)!r})'
        assert loss in ['l1', 'l2'], f'{loss!r} loss invalid.'
        if os.path.exists(out_path):
            assert os.path.isdir(out_path)
        else:
            os.makedirs(out_path)
        
        self.opt = opt.lower()
        self.input_size = 64 if self.opt == 'voxels-use64gan' else 32
        self.gan = self.opt[-3:] == 'gan'
        self.model_type = self.opt[:-3] if self.gan else self.opt
        self.data_file = data_file
        self.config_file = config_file
        self.evaluate_mode = evaluate_mode
        if data_file is not None:
            self.check_data_file()
        
        self.out_path = out_path
        self.num_workers = multiprocessing.cpu_count() if num_workers == -1 else num_workers
        self.batch_size = batch_size
        self.training_ratio = training_ratio
        self.minibatches_size = batch_size * training_ratio if self.gan else batch_size
        self.gradient_penalty = gradient_penalty
        self.loss = loss.lower()
        self.loss_multiply = loss_multiply
        self.data_loaded = False
        self.models_loaded = False
        self.training_vars_loaded = False
        
        # model files:
        self.generator_log_file = os.path.join(out_path, 'generator_log.csv')
        self.training_log_file = os.path.join(out_path, 'training_log.csv')
        self.model_g_file = os.path.join(out_path, 'model_generator.h5')
        self.active_files = [self.generator_log_file, self.training_log_file, self.model_g_file]
        if self.gan:
            self.discriminator_log_file = os.path.join(out_path, 'discriminator_log.csv')
            self.model_d_file = os.path.join(out_path, 'model_discriminator.h5')
            self.active_files += [self.discriminator_log_file, self.model_d_file]
            
        self.overwrite = overwrite
        self.trained = False
        self.check_training_files()
        self.n_labels = None
    
    def check_training_files(self):
        if self.overwrite:
            print('Cleaning files...')
            for i in [os.path.join(self.out_path, f) for f in os.listdir(self.out_path)]:
                if os.path.isfile(i):
                    os.remove(i)
        else:
            exists = []
            
            for f in self.active_files:
                if os.path.exists(f):
                    print(f'Found {f!r}')
                    exists.append(1)
                else:
                    exists.append(0)
            
            if np.sum(exists) > 0:
                self.trained = True
                for i,x in enumerate(exists):
                    if x == 0:
                        print(f'{self.active_files[i]!r} not found.')
                
                if np.sum(exists) != len(exists):
                    raise Exception('Some files not founded.')
            
    def check_data_file(self):
        if os.path.exists(self.data_file) and self.data_file[-4:] == '.npy':
            return
        
        with open(self.config_file) as f:
            for line in f.readlines():
                name, path = [l.strip() for l in line.split(':')[:2]]
                if name == self.data_file:
                    self.data_file = path
                    return
            
            raise Exception(f'{self.data_file!r} not found in {self.config_file!r}.')
    
    def load_data(self, force=False):
        if self.data_loaded and not force:
            print('Data already loaded (try force=True)')
            return
            
        loader = CustomLoader(self.data_file, self.out_path, 'train')
        self.len_train = len(loader)
        
        if self.gan:
            self.d_loader = torch.utils.data.DataLoader(loader, self.minibatches_size, shuffle=True,
                                                        num_workers=self.num_workers)
            
        self.g_loader = torch.utils.data.DataLoader(loader, self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers)
        self._load_n_labels()
        self.data_loaded = True
    
    def _load_n_labels(self):
        self.label_encoder = pickle.load(open(os.path.join(self.out_path, 'label_encoder.pkl'), 'rb'))
        self.n_labels = len(self.label_encoder.classes_)

    def build_models(self, force=False):
        if self.models_loaded and not force:
            print('Models already loaded (try force=True)')
            return
        
        if self.n_labels is None:
            self._load_n_labels()

        generator = make_generator(self.n_labels, self.model_type)
        self.generator = generator
        
        if self.gan and not self.evaluate_mode:
            discriminator = make_discriminator(self.n_labels, self.model_type)
            discriminator_model, generator_model = build_gan_arch(discriminator, generator,
                                                                  self.batch_size, self.gradient_penalty,
                                                                  self.loss, self.loss_multiply, self.input_size)
            self.discriminator = discriminator
            self.discriminator_model = discriminator_model
            self.generator_model = generator_model
        else:
            self.generator.compile('adam', 'mae')
            
        self.models_loaded = True
        
        if self.trained:
            self.load_weights()
        
    def init_training_vars(self, force=False):
        if self.training_vars_loaded and not force:
            print('Variables already loaded (try force=True)')
            return
        
        # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
        # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
        # gradient_penalty loss function and is not used.
        if self.gan:
            positive_y = np.ones((self.batch_size, 1), dtype=np.float32)
            negative_y = -positive_y
            dummy_y = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.training_vars = positive_y, negative_y, dummy_y
        
        self._load_full_test_set()
        self._load_full_train_set()
        self.training_vars_loaded = True
        
    def _load_full_train_set(self, **kwargs):
        data_train = CustomLoader(self.data_file, self.out_path, subset='train', **kwargs)
        len_train = len(data_train)
        train_loader = torch.utils.data.DataLoader(data_train, len_train, shuffle=True,
                                                         num_workers=self.num_workers)
        self.full_train_data = get_batch(iter(train_loader))
        # self.full_train_data = train_voxels, train_voxels_target, train_labels

    def _load_full_test_set(self, **kwargs):
        data_test = CustomLoader(self.data_file, self.out_path, subset='test', **kwargs)
        len_test = len(data_test)
        test_loader = torch.utils.data.DataLoader(data_test, len_test, shuffle=True,
                                                  num_workers=self.num_workers)
        self.full_test_data = get_batch(iter(test_loader))
        # self.full_test_data = test_voxels, test_voxels_target, test_labels

    def train(self, epochs, test_each=5, extend_training=True, max_extend=10):
        '''
        epochs: int
        test_each: logs loss on full test and train set each x epochs.
                   disabled on test_each=-1
        extend_training: bool, if True, the model will train for additional
                         epochs to ensure optimal model output.
        max_extend: maximum number of iterations to do in the extended train.
            
        '''
        if not self.data_loaded:
            print('Loading data...')
            self.load_data()
            
        if not self.models_loaded:
            print('Loading models...')
            self.build_models()
            
        if not self.training_vars_loaded:
            print('Loading training vars...')
            self.init_training_vars()
        
        print(f'\nDoing {epochs} epochs [{self.opt}]:')

        t0 = time()
        first_epoch = check_epoch(self.generator_log_file)
        epoch_range = range(first_epoch, first_epoch + epochs)
        iters = int(self.len_train // self.minibatches_size)
        left_epochs = epochs
        eta_hist = deque(maxlen = test_each if test_each > 0 else 5)
        test_hist = []
        extra = ''
        
        for epoch in epoch_range:
            eta_epoch = time()
            self._train_batch(iters, eta_hist, first_epoch, left_epochs, epoch, epochs)
            tested = False
            
            if test_each != -1 and (epoch == 0 or (epoch+1) % test_each == 0):
                train_loss, _ = self._evaluate_and_log(epoch)
                test_hist.append(train_loss)
                tested = True

            self.save_weights()
            left_epochs -= 1
            eta_hist.append(time() - eta_epoch)
        
        if extend_training:
            if not tested:
                train_loss, _ = self._evaluate_and_log(epoch)
                test_hist.append(train_loss)

            i = 0
            while i < max_extend:
                if np.min(test_hist) >= test_hist[-1] - 1e-4:
                    break
                else:
                    epoch += 1
                    self._train_batch(iters, eta_hist, first_epoch, 1, epoch, epochs)
                    train_loss, _ = self._evaluate_and_log(epoch)
                    test_hist.append(train_loss)
                    self.save_weights()

                i += 1
            else:
                extra += f'\nCould not achieve minimum after {max_extend} extra epochs.'
        
        print(f'Epoch: {epoch+1}/{first_epoch+epochs} | {i+1:02}/{iters:02}                   ')
        print(f'\nDone [{format_time(time() - t0)}]{extra}')
        print(f'Results on {self.out_path!r}')
    
    def _train_batch(self, iters, eta_hist, first_epoch, left_epochs, epoch, epochs):
        bs = self.batch_size
        iter_loader_g = iter(self.g_loader)

        if self.gan:
            # GAN Training
            iter_loader_d = iter(self.d_loader)
            positive_y, negative_y, dummy_y = self.training_vars
            for i in range(iters):
                eta = format_time(np.mean(eta_hist)*left_epochs) if len(eta_hist) > 0 else '---'
                print(f'Epoch: {epoch+1}/{first_epoch+epochs} (ETA: {eta}) | {i+1:02}/{iters:02}', end='\r')
                voxels_mb, voxels_target_mb, labels_mb = get_batch(iter_loader_d)
                for j in range(self.training_ratio):
                    voxels_batch = voxels_mb[j * bs:(j + 1) * bs]
                    voxels_target_batch = voxels_target_mb[j * bs:(j + 1) * bs]
                    labels_batch = labels_mb[j * bs:(j + 1) * bs]
                    discriminator_loss = self.discriminator_model.train_on_batch(
                        [voxels_target_batch, labels_batch, voxels_batch], [positive_y, negative_y, dummy_y])

                voxels_batch, voxels_target_batch, labels_batch = get_batch(iter_loader_g)
                generator_loss = self.generator_model.train_on_batch(
                    [voxels_batch, labels_batch], [positive_y, voxels_target_batch])

                write_discriminator_loss(self.discriminator_log_file, epoch+1, discriminator_loss)
                write_generator_loss(self.generator_log_file, epoch+1, generator_loss)
        else:
            # No GAN Training
            for i in range(iters):
                eta = format_time(np.mean(eta_hist)*left_epochs) if len(eta_hist) > 0 else '---'
                print(f'Epoch: {epoch+1}/{first_epoch+epochs} (ETA: {eta}) | {i+1:02}/{iters:02}', end='\r')
                voxels_batch, voxels_target_batch, labels_batch = get_batch(iter_loader_g)
                generator_loss = self.generator.train_on_batch([voxels_batch, labels_batch], voxels_target_batch)
                write_generator_loss(self.generator_log_file, epoch+1, generator_loss, gan=False)
    
    def _evaluate(self, dataset):
        '''return L1 loss'''
        voxels, voxels_target, labels = dataset
        result = self.generator.predict([voxels, labels], self.batch_size)
        loss = np.mean(np.abs(voxels_target - result))

        return loss

    def _evaluate_and_log(self, epoch):
        train_loss = self._evaluate(self.full_train_data)
        test_loss = self._evaluate(self.full_test_data)
        write_training_log(self.training_log_file, epoch+1, [train_loss, test_loss])

        return train_loss, test_loss

    def save_weights(self):
        if self.gan and not self.evaluate_mode:
            self.discriminator.save_weights(self.model_d_file)
            
        self.generator.save_weights(self.model_g_file)

    def load_weights(self):
        if self.gan and not self.evaluate_mode:
            self.discriminator.load_weights(self.model_d_file)
            
        self.generator.load_weights(self.model_g_file)
        print('Loaded weights from files')
        
    def format_input(self, voxels):
        '''from binary to [-1,1] space'''
        if voxels.dtype == np.bool:
            voxels = voxels.astype(np.int)

        idxs = np.argwhere(voxels == 0)
        if len(voxels.shape) == 3:
            voxels[idxs[:,0], idxs[:,1], idxs[:,2]] = -1
        elif len(voxels.shape) == 4:
            voxels[idxs[:,0], idxs[:,1], idxs[:,2], idxs[:,3]] = -1
        else:
            raise Exception('Invalid shape.')

        return voxels

    def format_output(self, voxels):
        '''from [-1,1] to binary space'''
        return voxels > 0
    
    def predict(self, batch_voxels, batch_labels, decode_label=False,
                format_input=False, format_output=False):
        '''
        format_input: from binary to [-1,1] space
        format_output: from [-1,1] to binary space
        '''
        if not self.models_loaded:
            print('Loading models...')
            self.build_models()

        if decode_label:
            batch_labels = self.label_encoder.transform(batch_labels)

        if format_input:
            batch_voxels = self.format_input(batch_voxels)

        result = self.generator.predict([batch_voxels, batch_labels], self.batch_size)
        result[batch_voxels == 1] = 1

        if format_output:
            result = self.format_output(result)

        return result

    def predict_one(self, voxels, label, decode_label=False,
                    format_input=False, format_output=False):
        '''
        format_input: from binary to [-1,1] space
        format_output: from [-1,1] to binary space
        '''
        if not self.models_loaded:
            print('Loading models...')
            self.build_models()

        if decode_label:
            label = self.label_encoder.transform([label])
        else:
            label = [label]

        if len(voxels.shape) == 3:
            voxels = np.expand_dims(voxels, 0)

        if format_input:
            voxels = self.format_input(voxels)

        result = self.generator.predict([voxels, label])
        result[voxels == 1] = 1

        if format_output:
            result = self.format_output(result)

        return result

