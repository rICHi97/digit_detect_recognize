import os
from tensorflow import Session, logging, ConfigProto
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,TensorBoard,LearningRateScheduler
from keras.optimizers import Adam

# 直接运行，改为绝对导入
# import east.net.cfg as cfg
# from east.net.network import East
# from east.net.losses import quad_loss
# from east.net.data_generator import gen

from . import cfg as cfg
from .network import East
from .losses import quad_loss
from .data_generator import gen

# import cfg as cfg
# from network import East
# from losses import quad_loss
# from data_generator import gen

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
# 只显示error
os.environ['TF_MIN_CPP_LOG_LEVEL']='2'
logging.set_verbosity(logging.ERROR)
config=ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth=True
session=Session(config=config)

east = East()
east_network = east.east_network()
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=Adam(lr = cfg.lr,
                                                    # clipvalue=cfg.clipvalue,
                                                    decay=cfg.decay
                                                    ))
def scheduler(epoch, lr):
    lr = lr * 10
    return lr

# saved_model_weights_file_path = 'east_model_weights_%s.h5'% train_task_id
# if cfg.load_weights: # and os.path.exists(cfg.saved_model_weights_file_path):
#    east_network.load_weights( 'pre_train_val_loss_0.376.h5')


early_stopping = EarlyStopping(patience= 6, verbose=1)
check_point = ModelCheckpoint(filepath=cfg.model_weights_path, save_best_only=True, save_weights_only=True, verbose=1, period= 1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 6, verbose=1, min_lr = 1e-6)
tb = TensorBoard(log_dir = 'model/logs', histogram_freq=0, update_freq = 'epoch')
lr_scheduler = LearningRateScheduler(scheduler)

east_network.fit_generator(generator = gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs= cfg.epoch_num,
                           # 通过再次调用生成验证集数据
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           # callbacks=[early_stopping, check_point, reduce_lr]
                           callbacks=[check_point, reduce_lr])
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)


# yaml_string = east_network.to_yaml()
# with open("east_model.yaml", "w") as f:
#     f.write(yaml_string)