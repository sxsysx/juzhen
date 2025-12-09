 Tensorboard 可视化使用指南


 1. 单优化器训练可视化

 1.1 运行训练并生成日志

使用以下命令运行训练，会自动生成Tensorboard日志：

```bash
python main.py --model vgg11 --dataset stl10 --optimizer sgd --batch_size 128 --epochs 100
```

日志将保存在 `tensorboard_logs/` 目录下，文件夹名称包含了训练配置信息。

 1.2 查看Tensorboard可视化结果

训练过程中或训练完成后，可以使用以下命令启动Tensorboard：

```bash
tensorboard --logdir=tensorboard_logs
```

然后在浏览器中访问 http://localhost:6006 查看可视化结果。

2. 多优化器结果比较

2.1 同时训练多个优化器

使用 `--compare_optimizers` 参数和 `--optimizer_list` 参数可以同时训练多个优化器并在Tensorboard中比较它们的性能：

```bash
python main.py --model vgg11 --dataset stl10 --compare_optimizers --optimizer_list sgd,sgd_momentum,adam,adamw,nadam,rmsprop,adagrad --batch_size 128 --epochs 50
```

2.2 比较多个优化器的结果

使用以下命令启动Tensorboard来比较多个优化器的结果：

```bash
tensorboard --logdir=tensorboard_logs/comparison
```

在Tensorboard界面中，相同类型的指标（如损失、准确率）将自动在同一张图表上显示，不同的线代表不同的优化器。

3. 可视化指标说明

在Tensorboard中，您可以查看以下指标：

- Loss/train: 训练损失值
- Loss/test: 测试损失值
- Error/train: 训练错误率
- Error/test: 测试错误率
- Accuracy/test: 测试准确率
- Learning Rate: 学习率变化

4. 支持的优化器

现在支持以下优化器进行训练和比较：

- SGD (Stochastic Gradient Descent with Nesterov momentum)
- SGD with Momentum (without Nesterov)
- Adam
- AdamW (Adam with weight decay regularization)
- Nadam (Adam with Nesterov momentum approximation)
- RMSprop
- Adagrad

5. 示例命令

5.1 快速测试（较少epochs）
如果你想要快速测试多优化器比较功能，可以使用较小的批次大小和较少的训练轮次：

```bash
python main.py --model vgg11 --dataset stl10 --compare_optimizers --optimizer_list sgd,sgd_momentum,adam,adamw --batch_size 32 --epochs 5
```

5.2 完整训练
对于更复杂的比较，你可以尝试不同的模型和数据集：

```bash
python main.py --model resnet18 --dataset stl10 --compare_optimizers --optimizer_list sgd,sgd_momentum,adam,adamw,nadam --batch_size 128 --epochs 200 --lr 0.01
```

