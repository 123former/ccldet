#CUDA_VISIBLE_DEVICES=1 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains/1A/20230829028/ work_dirs/faster_rcnn_r50_fpn_all_expand/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all_expand/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all_expand/vis3/1A/

#CUDA_VISIBLE_DEVICES=3 python structure.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_image_split/下载科佳fix图/20230323001 /home/f523/guazai/disk3/shangxiping/mmrotate/model_sxp_576_576_112.onnx --out-file vis1102/3C/
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CRH6F/20231005013 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/CRH6F/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CRH2E/20231012015 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/2E/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CRH380AL/20231007019 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/CRH380AL/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CR400AF-A/20231007014 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/400AF-A/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CR400AF-Z/20230723106 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/CR400AF-Z/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CR400BF/20231005020 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_addvis/CR400BF/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CR400BF-Z/20231006014 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/CR400BF-Z/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CRH2A/20231004023 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/CRH2A/
#
#CUDA_VISIBLE_DEVICES=2 python demo/image_demo_hbb.py /home/f523/guazai/sdb/故障检测/dataset_trains_1.5/all_bg_image_split/CRH3C/20230619027 work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/epoch_12.pth --out-file work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/vis/3C/

#python tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs/faster_rcnn_r50_fpn_all+aug_large_add/faster_rcnn_r50_fpn.py
#
#python demo/test_and_save_with_json.py

#sudo shutdown -h 60
#python ./tools/dc_tools/make_dataset.py
#python ./tools/dc_tools/FAIR1M2COCO_v2.py
#python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs/faster_rcnn_r50_fpn_all+aug_sample/faster_rcnn_r50_fpn.py
#
#python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs_response/faster_rcnn_llvip_rgb/faster_rcnn_r50_fpn.py

python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs_response/faster_rcnn_llvip_rgb_vis/faster_rcnn_r50_fpn.py

python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs_response/faster_rcnn_llvip_rgb_vis_mt/faster_rcnn_r50_fpn.py

python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs_response/faster_rcnn_llvip_rgb_vis_mt2/faster_rcnn_r50_fpn.py

python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs_response/faster_rcnn_llvip_rgb_vis_mt3/faster_rcnn_r50_fpn.py

python ./tools/train.py /home/f523/guazai/disk3/shangxiping/mmrotate/work_dirs_response/faster_rcnn_llvip_rgb_vis_mt4/faster_rcnn_r50_fpn.py