bash tools/dist_test.sh \
    configs/deformable_detr/decoderAug/decoderAug_group5_t1500_deformable_detr_twostage_refine_r50_16x2_12e_coco.py \
    /openseg_blob_new/jiading/outputs-aml/DecoderAug-mmdetection-v2-6-15/deformable_detr/decoderAug/decoderAug_group5_t1500_deformable_detr_twostage_refine_r50_16x2_12e_coco/epoch_2.pth \
    4 \
    --eval bbox