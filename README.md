# YOLO_V3
所有文件请从https://github.com/qqwweee/keras-yolo3 下载，本库内文件已重新训练，针对新数据集进行了模型训练，加入了对各个子文件的解释
关于yolo环境的建立
关于yolo的运行
关于通过yolo训练自己的模型

![yolov3主体框架](https://github.com/Tu-Jiawei/YOLO_V3/blob/master/test_yolo_v3/2018100917221176.jpg)

coco_classes:包含coco数据集内的80个对象名称，对自己的数据集进行训练时需要修改，格式参考本版本

voc_classes:包含部分coco数据集内的对象，对自己的数据集进行训练时需要修改，格式参考本版本

yolo_anchors:是yolo作者经大量测试得到的9个anchors，我理解为框，在输入数据库内的图片经预处理转换为yolo所需格式后（一般为416
