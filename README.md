# YOLO_V3
所有文件请从https://github.com/qqwweee/keras-yolo3 下载，本库内文件已重新训练，针对新数据集进行了模型训练，加入了对各个子文件的解释
关于yolo环境的建立

关于yolo的运行

关于通过yolo训练自己的模型

weights文件是caffe常用的模型参数存储文件，在tf环境下使用时需要先转换为.H5格式的文件

![yolov3主体框架](https://github.com/Tu-Jiawei/YOLO_V3/blob/master/test_yolo_v3/2018100917221176.jpg)

coco_classes:包含coco数据集内的80个对象名称，对自己的数据集进行训练时需要修改，格式参考本版本

voc_classes:包含部分coco数据集内的对象，对自己的数据集进行训练时需要修改，格式参考本版本

yolo_anchors:是yolo作者经大量测试得到的9个anchors，我理解为框，在输入数据库内的图片经预处理转换为yolo所需格式后（一般为416*416），此时如anchor（10,13），指的是以10*13的框为对象选择框，在图片上进行查找，当然不是固定的，经训练后产生的whxy值便是对这个anchor框的微调，目的永远是使最终的iou值越大越好

convert:实现caffe的Weights格式到tf的.H5格式的转换
