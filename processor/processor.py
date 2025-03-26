import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             query_loader, gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(cfg, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_rank = 0
    best_rank2 = 0
    best_rank3 = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()

        for n_iter, (img, vid, target_cam, path, item) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            item = item.to(device)## modality id
            # target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                
                score, feat = model(img, target, cam_label=target_cam, view_label=item)
                target = torch.cat((target[item==1], target[item==0]), dim=0)
                item = torch.cat((item[item==1], item[item==0]), dim=0)

                loss = loss_fn(score, feat, target, target_cam, item)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    
                    for n_iter, (img, vid, camid, _,  _) in enumerate(query_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camid = camid.to(device)
                            # target_view = target_view.to(device)
                            feat = model(img, cam_label=camid)
                            evaluator.update((feat, vid, camid))
                    for n_iter, (img, vid, camid, _, _) in enumerate(gallery_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camid = camid.to(device)
                            # target_view = target_view.to(device)
                            feat = model(img, cam_label=camid)
                            evaluator.update((feat, vid, camid))
                    
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, path, item) in enumerate(query_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camid = camid.to(device)
                        item = item.to(device)
                        # target_view = target_view.to(device)
                        gfeat, sfeat, feat = model(img, cam_label=camid, view_label=item, mode=0)#, view_label=target_view)
                        evaluator.update((gfeat, sfeat, feat, vid, camid, path))
                
                for n_iter, (img, vid, camid, path, item) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camid = camid.to(device)
                        # target_view = target_view.to(device)
                        gfeat, sfeat, feat = model(img, cam_label=camid, view_label=item, mode=1)
                        evaluator.update((gfeat, sfeat, feat, vid, camid, path))

                cmc0, mAP0, cmc, mAP, cmc1, mAP1 = evaluator.compute()
                # logger.info("Validation Results - Epoch: {}".format(epoch))

                # logger.info("mAP: {:.1%}".format(mAP0))
                # for r in [1, 5, 10]:
                #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc0[r - 1]))

                # logger.info("BN mAP: {:.1%}".format(mAP))
                # for r in [1, 5, 10]:
                #     logger.info("BN CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                if cmc0 > best_rank:
                    best_rank = cmc0
                    torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_global_bn_best.pth'))
                if cmc > best_rank2:
                    best_rank2 = cmc
                    torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_specific__bn_best.pth'))
                
                if cmc1 > best_rank3:
                    best_rank3 = cmc1
                    torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_fusion_bn_best.pth'))

                    
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 query_loader,
                 gallery_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(cfg, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, imgpath, item) in enumerate(query_loader):
        with torch.no_grad():
            img = img.to(device)
            camid = camid.to(device)
            # target_view = target_view.to(device)
            gfeat, sfeat, feat = model(img, cam_label=camid, view_label=item, mode=0)
            evaluator.update((gfeat, sfeat, feat, pid, camid, imgpath))

            # img_path_list.extend(imgpath)
    
    for n_iter, (img, pid, camid, imgpath, item) in enumerate(gallery_loader):
        with torch.no_grad():
            img = img.to(device)
            camid = camid.to(device)
            # target_view = target_view.to(device)
            gfeat, sfeat, feat = model(img, cam_label=camid, view_label=item, mode=1)
            evaluator.update((gfeat, sfeat, feat, pid, camid, imgpath))
            # img_path_list.extend(imgpath)

    cmc0, mAP0, cmc, mAP, cmc1, mAP1  = evaluator.compute()
    # logger.info("Validation Results ")
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    return cmc0, cmc, cmc1


