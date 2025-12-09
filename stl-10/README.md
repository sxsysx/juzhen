1.训练参数

--dataset：选择数据集，stl10

--data_dir：数据集根目录文件夹路径（默认：~/dataset）

--batch_size：批量大小（默认：128）

--lr：学习率（默认：0.1）

--lr_decay：学习率衰减率（默认：0.1）

--optimizer：选择优化器（默认：sgd）

--weight_decay：权重衰减（默认：0.0005）

--momentum：动量（默认：0.9）

--epochs：训练总轮数（默认：300）

--save：保存训练后模型的路径（默认：trained_nets）

--save_epoch：每隔多少轮保存一次（默认：10）

--ngpu：使用的 GPU 数量（默认：1）

--rand_seed：随机数生成器种子（默认：0）

--resume_model：从检查点恢复模型（默认：''）

--resume_opt：从检查点恢复优化器（默认：''）

2.模型参数

--model，-m：模型架构（仅可选 vgg11、16、19，resnet18、34）（默认：resnet18）

--loss_name：选择损失函数，可选 crossentropy 和 mse（默认：'crossentropy'）

3.数据参数

--raw_data：不进行数据标准化（默认：False）

--noaug：不使用数据增强（默认：False）

--label_corrupt_prob：标签噪声概率（默认：0.0）

--trainloader：包含随机标签的训练数据加载器路径（默认：''）

--testloader：包含随机标签的测试数据加载器路径（默认：''）
