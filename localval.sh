pwd
nvidia-smi
GPUS_PER_NODE=2 tools/dist_test.sh \
    Projects/configs_plugin/checkpoint_deformable_detr/checkpoint_deformable_detr_twostage_refine_r50_dim2048_16x2_36e_coco.py \
    /openseg_blob_new/v-yiduohao/pretrained_model/hdetr-mmdet/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
    2 --eval bbox \
    --work-dir /openseg_blob_new/v-yiduohao/outputs-aml/playground