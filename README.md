# MYAI
mutil_label NER1
当前模型存在的问题主要有两点：
1、部分实体的引入会造成大量的空间开销
2、不可避免的暴露偏差，特指对于非实体的判别（也就是对模型精度的研磨，对非实体直接设置结束符可能丢失了对类似实体的非法序列的判别）也即对非实体的判断问题有待提高
3、标签信息没能引入，单单引入标签的编码（随机初始化）。
