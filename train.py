from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer


config = Cfg.load_config_from_name('vgg_seq2seq')


dataset_params = {
    'name': 'hw',
    'data_root': r'G:\Text Recognition 1\dataset 6',  
    'train_annotation': r'G:\Text Recognition 1\dataset 6\train\train.txt', 
    'valid_annotation': r'G:\Text Recognition 1\dataset 6\val\val.txt'   
}


params = {
    'print_every': 50,  
    'valid_every': 500,  
    'iters': 10000,  
    'checkpoint': r'G:\Text Recognition 1\checkpoint\checkpoint.pth',  
    'export': r'G:\Text Recognition 1\weight\seq2seqocr1.pth',  
    'metrics': 500 
}


config['trainer'].update(params)
config['dataset'].update(dataset_params)

config['trainer']['batch_size'] = 16
config['trainer']['lr'] = 1e-4
config['trainer']['print_loss'] = True

config['device'] = 'cuda'
config['dataloader']['num_workers'] = 0

config['dataset']['image_max_width'] = 256
config['dataset']['image_max_height'] = 32

# Khởi tạo Trainer với mô hình đã cấu hình
trainer = Trainer(config, pretrained=True)

# Bắt đầu quá trình huấn luyện
trainer.train()
