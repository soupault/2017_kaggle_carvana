from common import *

from dataset.carvana_cars import *
from net.segmentation.my_unet import DiceLoss, BCELoss2d, UNet128_1 as Net
from net.tool import *


## experiment setting here ----------------------------------------------------
def criterion(logits, labels):
    l = BCELoss2d()(logits, labels) # + DiceLoss()(logits, labels)
    return l



## experiment setting here ----------------------------------------------------




#https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
#https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score

#https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2 ):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = score.sum()/num
    return score




def predict(net, test_loader):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_H, CARVANA_W
    predictions  = np.zeros((num, H, W),np.float32)

    test_num  = 0
    for it, (images, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)

        batch_size = len(indices)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1, H, W)

    assert(test_num == len(test_loader.sampler))
    return predictions



def predict_and_evaluate(net, test_loader ):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_H, CARVANA_W
    predictions  = np.zeros((num, H, W),np.float32)

    test_acc  = 0
    test_loss = 0
    test_num  = 0
    for it, (images, labels, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)
        labels = Variable(labels.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)
        masks  = (probs>0.5).float()

        loss = criterion(logits, labels)
        acc  = dice_loss(masks, labels)


        batch_size = len(indices)
        test_num  += batch_size
        test_loss += batch_size*loss.data[0]
        test_acc  += batch_size*acc.data[0]
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1, H, W)

    assert(test_num == len(test_loader.sampler))

    test_loss = test_loss/test_num
    test_acc  = test_acc/test_num

    return predictions, test_loss, test_acc










def show_train_batch_results(probs, labels, images, indices, wait=1, save_dir=None, names=None):

    probs  = (probs.data.cpu().numpy().squeeze()*255).astype(np.uint8)
    labels = (labels.data.cpu().numpy()*255).astype(np.uint8)
    images = (images.data.cpu().numpy()*255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))

    batch_size,H,W,C = images.shape
    results = np.zeros((H, 3*W, 3),np.uint8)
    prob    = np.zeros((H, W, 3),np.uint8)
    for b in range(batch_size):
        m = probs [b]>128
        l = labels[b]>128
        score = one_dice_loss_py(m , l)

        image = images[b]
        prob[:,:,1] = probs [b]
        prob[:,:,2] = labels[b]

        results[:,  0:W  ] = image
        results[:,  W:2*W] = prob
        results[:,2*W:3*W] = cv2.addWeighted(image, 1, prob, 1., 0.) # image * α + mask * β + λ
        draw_shadow_text  (results, '%0.3f'%score, (5,15),  0.5, (255,255,255), 1)

        if save_dir is not None:
            shortname = names[indices[b]].split('/')[-1].replace('.jpg','')
            cv2.imwrite(save_dir + '/%s.jpg'%shortname, results)


        im_show('train',  results,  resize=1)
        cv2.waitKey(wait)



# ------------------------------------------------------------------------------------
def run_train():


    out_dir  = '/root/share/project/kaggle-carvana-cars/results/xx2-23'

    #logging, etc --------------------
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir+'/train/results', exist_ok=True)
    os.makedirs(out_dir+'/valid/results', exist_ok=True)
    os.makedirs(out_dir+'/test/results',  exist_ok=True)
    os.makedirs(out_dir+'/backup', exist_ok=True)
    os.makedirs(out_dir+'/checkpoint', exist_ok=True)
    os.makedirs(out_dir+'/snap', exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/train.code.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')




    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 32
    train_dataset = KgCarDataset( 'train128x128_v0_4320',    #'train128x128_5088',  #'train_5088'
                                transform=[
                                    lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(-0,0)),
                                    lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
                                is_label=True,
                                is_preload=True,)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),  #ProbSampler(train_dataset),  #ProbSampler(train_dataset,SAMPLING_PROB),  # #FixedSampler(train_dataset,list(range(batch_size))),  ##
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)



    valid_dataset = KgCarDataset( 'valid128x128_v0_768',
                                is_label=True,
                                is_preload=True,)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 3,
                        pin_memory  = True)


    test_dataset = KgCarDataset( 'test128x128_3197',
                                  is_label=False,
                                  is_preload=False,)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)

    H,W = CARVANA_H, CARVANA_W

    ## net ----------------------------------------
    log.write('** net setting **\n')

    net = Net(in_shape=(3, H, W), num_classes=1)
    net.cuda()

    log.write('%s\n\n'%(type(net)))

    ## optimiser ----------------------------------
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005

    num_epoches = 23  #100
    it_print    = 1   #20
    it_smooth   = 20
    epoch_test  = 1
    epoch_valid = 1
    epoch_save  = [0,3,5,10,15,20,25,35,40,45,50, num_epoches-1]

    ## resume from previous ----------------------------------
    start_epoch=0

    #training ####################################################################3
    log.write('** start training here! **\n')
    log.write('\n')


    log.write('epoch    iter      rate   | smooth_loss/acc | train_loss/acc | test_loss/acc ... \n')
    log.write('--------------------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    smooth_acc  = 0.0
    train_loss = np.nan
    train_acc  = np.nan
    valid_loss = np.nan
    valid_acc  = np.nan
    time = 0
    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        #print ('epoch=%d'%epoch)
        start = timer()

        #---learning rate schduler ------------------------------
        # lr = LR.get_rate(epoch, num_epoches)
        # if lr<0 : break
        if epoch>=20:
            adjust_learning_rate(optimizer, lr=0.001)

        rate =  get_learning_rate(optimizer)[0] #check
        #--------------------------------------------------------


        sum_smooth_loss = 0.0
        sum_smooth_acc  = 0.0
        sum = 0
        net.train()
        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):
            images  = Variable(images.cuda())
            labels  = Variable(labels.cuda())

            #forward
            logits = net(images)
            probs = F.sigmoid(logits)
            masks = (probs>0.5).float()


            #backward
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            acc  = dice_loss(masks, labels)

            sum_smooth_loss += loss.data[0]
            sum_smooth_acc  += acc .data[0]
            sum += 1

            if it%it_smooth == 0:
                smooth_loss = sum_smooth_loss/sum
                smooth_acc  = sum_smooth_acc /sum
                sum_smooth_loss = 0.0
                sum_smooth_acc  = 0.0
                sum = 0


            if it%it_print == 0 or it==num_its-1:
                train_acc  = acc.data [0]
                train_loss = loss.data[0]
                print('\r%5.1f   %5d    %0.4f   |  %0.4f  %0.4f | %0.4f  %6.4f | ... ' % \
                        (epoch + (it+1)/num_its, it+1, rate, smooth_loss, smooth_acc, train_loss, train_acc),\
                        end='',flush=True)


            #debug show prediction results ---
            #if 1:
                show_train_batch_results(probs, labels, images, indices,
                                         wait=1, save_dir=out_dir+'/train/results', names=train_dataset.names)

        end  = timer()
        time = (end - start)/60
        #end of epoch --------------------------------------------------------------



        if epoch % epoch_valid == 0 or epoch == 0 or epoch == num_epoches-1:
            net.eval()
            valid_predictions, valid_loss, valid_acc = predict_and_evaluate(net, valid_loader)


            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.4f  %0.4f | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, smooth_acc, train_loss, train_acc, valid_loss, valid_acc, time))

        if 1:
        #if epoch % epoch_test == 0 or epoch == 0 or epoch == num_epoches-1:
            net.eval()
            probs = predict(net, test_loader)

            results = np.zeros((H, 3*W, 3),np.uint8)
            prob    = np.zeros((H, W, 3),np.uint8)
            num_test = len(probs)
            for b in range(100):
                n = random.randint(0,num_test-1)
                shortname    = test_dataset.names[b].split('/')[-1].replace('.jpg','')
                image, index = test_dataset[n]
                image        = tensor_to_image(image, std=255)
                prob[:,:,1]  = probs[n]*255

                results[:,  0:W  ] = image
                results[:,  W:2*W] = prob
                results[:,2*W:3*W] = cv2.addWeighted(image, 1, prob, 1., 0.) # image * α + mask * β + λ

                cv2.imwrite(out_dir+'/test/%s.jpg'%shortname, results)
                im_show('test',  results,  resize=1)
                cv2.waitKey(1)

        if epoch in epoch_save:
            torch.save(net.state_dict(),out_dir +'/snap/%03d.pth'%epoch)
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/checkpoint/%03d.pth'%epoch)
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py





    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-test
    torch.save(net.state_dict(),out_dir +'/snap/final.pth')


# ------------------------------------------------------------------------------------
# https://www.kaggle.com/tunguz/baseline-2-optimal-mask/code
def run_submit():

    out_dir  = '/root/share/project/kaggle-carvana-cars/results/xx6-128'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 64

    test_dataset = KgCarDataset( 'test256x256_100064',
                                  is_label=False,
                                  is_preload=False,  #True,
                               )
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)

    H,W = CARVANA_H, CARVANA_W

    ## net ----------------------------------------
    net = Net(in_shape=(3, H, W), num_classes=1)
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    ## start testing now #####
    log.write('start prediction ...\n')
    if 1:
        net.eval()
        probs = predict( net, test_loader )
        np.save(out_dir+'/submit/probs.npy', probs)
    else:
        probs = np.load(out_dir+'/submit/probs.npy')


    if 1:
        num = 500
        results = np.zeros((H, 3*W, 3),np.uint8)
        prob    = np.zeros((H, W, 3),np.uint8)
        num_test = len(probs)
        for n in range(num):
            shortname    = test_dataset.names[n].split('/')[-1].replace('.jpg','')
            image, index = test_dataset[n]
            image        = tensor_to_image(image, std=255)
            prob[:,:,1]  = probs[n]*255

            results[:,  0:W  ] = image
            results[:,  W:2*W] = prob
            results[:,2*W:3*W] = cv2.addWeighted(image, 0.75, prob, 1., 0.) # image * α + mask * β + λ

            cv2.imwrite(out_dir+'/submit/results/%s.jpg'%shortname, results)
            im_show('test',  results,  resize=1)
            cv2.waitKey(1)


    #resize to original and make csv

    print('make csv')
    threshold=0.5
    csv_file = out_dir+'/submit/results-th%05f.csv'%threshold
    gz_file = csv_file +'.gz'


    #verify file order is correct!
    num_test = len(test_dataset.names)
    names=[]
    for n in range(num_test):
        name = test_dataset.names[n].split('/')[-1]+'.jpg'
        names.append(name)

    start = timer()
    rles=[]
    for n in range(num_test):
        if (n%1000==0):
            end  = timer()
            time = (end - start) / 60
            time_remain = (num_test-n-1)*time/(n+1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))

        prob = probs[n]
        prob = cv2.resize(prob,(CARVANA_WIDTH,CARVANA_HEIGHT))
        mask = prob>threshold
        rle = run_length_encode(mask)
        rles.append(rle)

        #im_show('prob', prob*255, resize=0.333)
        #cv2.waitKey(0)

    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
    #run_submit()

    print('\nsucess!')