CUDA_VISIBLE_DEVICES='5' python train.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.AD True MODEL.WAD 0.1 MODEL.A1 0.0 OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/'

CUDA_VISIBLE_DEVICES='5' python test.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.AD True MODEL.WAD 0.1 MODEL.A1 0.0 MODEL.DEVICE_ID "'0'" OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/' TEST.WEIGHT '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/transformer_global_bn_best.pth'

CUDA_VISIBLE_DEVICES='5' python test.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.AD True MODEL.WAD 0.1 MODEL.A1 0.0 MODEL.DEVICE_ID "'0'" OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/' TEST.WEIGHT '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/transformer_specific__bn_best.pth'

CUDA_VISIBLE_DEVICES='5' python test.py --config_file configs/sysu/vit_base_ics_384.yml MODEL.AD True MODEL.WAD 0.1 MODEL.A1 0.0 MODEL.DEVICE_ID "'0'" OUTPUT_DIR '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/' TEST.WEIGHT '/data1/ccq/transreid_cross_modality/sysu/market_transreid_base_cls_384_crossfusiontokenadversiarladd01_0fusionprototypeitc/transformer_fusion_bn_best.pth'
