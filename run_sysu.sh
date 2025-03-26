CUDA_VISIBLE_DEVICES='0' python train.py --config_file configs/sysu/vit_base_ics_384.yml OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/'

CUDA_VISIBLE_DEVICES='0' python test.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.DEVICE_ID "'0'" OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/' TEST.WEIGHT '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/transformer_global_bn_best.pth'

CUDA_VISIBLE_DEVICES='0' python test.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.DEVICE_ID "'0'" OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/' TEST.WEIGHT '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/transformer_specific__bn_best.pth'

CUDA_VISIBLE_DEVICES='0' python test.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.DEVICE_ID "'0'" OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/' TEST.WEIGHT '/data1/ccq/transreid_cross_modality/sysu/lup_transreid_base_cls_384_crossfusiontoken_new2/transformer_fusion_bn_best.pth'
