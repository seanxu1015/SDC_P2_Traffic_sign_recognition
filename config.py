class Config:
    training_file = 'train.p'
    testing_file = 'test.p'
    learning_rate = 1e-5
    wd = 1e-5
    dropout = .5
    batch_size = 512
    seed = 161210
    val_size = 0.1
    epochs = 4000
    interval = 10
    logs_path = './save'
