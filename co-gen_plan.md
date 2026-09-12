请你仔细阅读这个repo, 帮我生成一份新的研究计划
/workspace/ttdu/ActionImages-Cogen

我想要把研究重点放在有多模态生成的能力上，所以基于官方的/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt
我只要有更多的模态生成能力，并不强求action 模态在close loop execution 上最好；


从这个代码出发/workspace/ttdu/ActionImages-Cogen/scripts/train_arm.sh
arm4 MIX="video+action@0.6,video+depth@0.4" ;;这个训练出来切换prompt 能学习到depth 视频，

然后可以做arm 1, 帮我修改arm 1
arm1) MIX="video+action@0.6,video+depth@0.2,video+segmentation@0.2" ;;

segmentation 之前做了一个新的计划和代码，除了目标物体其他物体也能seg 出来，我要一个全面的mask 图，帮我debug 确认这部分，生成例子看一下；
另外，参考这篇文章/workspace/ttdu/ttd/papers/Zhuang_Argus_A_Compact_and_Versatile_Foundation_Model_for_Vision_CVPR_2025_paper.pdf

有哪些模态可以增加？

有一个疑惑的是，要不要mix 3个dataset droid/bridge/rlbench, 只有rlbench 的多模态数据目前比较全面 
