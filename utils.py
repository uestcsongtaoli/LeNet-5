import os


def save_model(model, model_name, model_dir="models"):
    """
    保存模型
    :param model: 训练得到的模型
    :param model_name: 模型命名
    :param model_dir: 模型要存储的文件夹，默认是models
    :return: None
    """

    # 创建文件夹
    os.makedirs(model_dir, exist_ok=True)

    # 模型存储的文件路径
    save_path = os.path.join(model_dir, model_name)

    # 存储模型
    model.save(save_path)

    # 删除已经存储过的模型
    del model
    return

