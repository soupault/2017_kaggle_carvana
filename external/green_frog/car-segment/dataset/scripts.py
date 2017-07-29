#some usefule scripts
from common import *
from dataset.carvana_cars import *

#make resized images for faster processing
def run_make_small_images():

    img_dir  = CARVANA_DIR + '/images/train'

    H,W = 320, 320
    small_dir  = img_dir + '%dx%d'%(W,H)
    os.makedirs(small_dir,exist_ok=True)

    img_list = glob.glob(img_dir + '/*.jpg')
    num_imgs = len(img_list)
    for n in range(num_imgs):
        print('n/num_imgs=%d/%d'%(n,num_imgs))

        img_file = img_list[n]
        img = cv2.imread(img_file)
        img = cv2.resize(img,(W,H))
        save_file = img_file.replace(img_dir, small_dir)
        cv2.imwrite(save_file,img)

    pass


#make resized images for faster processing
def run_make_small_masks():

    img_dir = CARVANA_DIR + '/annotations/train'  # read all annotations

    H,W = 320, 320
    small_dir  = img_dir + '%dx%d'%(W,H)
    os.makedirs(small_dir,exist_ok=True)

    img_list = glob.glob(img_dir + '/*.gif')
    num_imgs = len(img_list)
    for n in range(num_imgs):
        print('n/num_imgs=%d/%d'%(n,num_imgs))

        img_file = img_list[n]
        img = PIL.Image.open(img_file)
        img = np.array(img)*255
        img = cv2.resize(img,(W,H))
        save_file = img_file.replace(img_dir, small_dir).replace('.gif', '.png')
        cv2.imwrite(save_file,img)

    pass



# make baseline submission by using average mask --------------------------------------
def run_make_baseline_submission():

    if 0:
        img_dir  = CARVANA_DIR + '/images/train'
        mask_dir = CARVANA_DIR + '/annotations/train_masks'  # read all annotations

        ave_dir  = CARVANA_DIR + '/others/ave'
        os.makedirs(ave_dir,exist_ok=True)

        for v in range(1,CARVANA_NUM_VIEWS+1):
            img_list = glob.glob(img_dir + '/*_%02d.jpg'%v)
            num_imgs = len(img_list)
            print('v=%02d : num_imgs=%d'%(v,num_imgs))
            for n in range(num_imgs):
                img_file = img_list[n]
                shortname = img_file.split('/')[-1].replace('.jpg','')
                mask_file = mask_dir + '/' + shortname + '_mask.gif'
                mask = PIL.Image.open(mask_file)
                mask = np.array(mask)

                if n==0:
                    average = np.zeros(mask.shape,np.float32)
                average += mask

            average = np.round(average/num_imgs)
            average = average.astype(np.uint8)

            im_show('average @ %d'%v, average*255, resize=0.25)
            cv2.imwrite(ave_dir + '/%02d.png'%v, average*255)
            cv2.waitKey(1)
    pass


    csv_file = '/root/share/project/kaggle-carvana-cars/results/submission/average-00.csv'
    zip_file = csv_file +'.zip'
    if 1:
        ## read average mask
        rles=['',]
        ave_dir  = CARVANA_DIR + '/others/ave'
        for v in range(1,CARVANA_NUM_VIEWS+1):
            img_file = ave_dir + '/%d.png'%v
            average = cv2.imread(ave_dir + '/%02d.png'%v, cv2.IMREAD_GRAYSCALE)/255
            rle = run_length_encode(average)
            #print(rle)

            rles.append(rle)
            im_show('average', average*255, resize=0.25)
            cv2.waitKey(1)


        # read names
        list = CARVANA_DIR +'/images/test.txt'
        with open(list) as f:
            shortnames = f.readlines()
        shortnames = [x.strip()for x in shortnames]
        num_test   = len(shortnames)

        with open(csv_file,'w') as f:
            f.write('img,rle_mask\n')
            for n in range(num_test):
                shortname = shortnames[n]
                v = int(shortname[-2:])
                f.write('%s.jpg,%s\n'%(shortname,rles[v]))

    print( 'convert to zip')
    zf = zipfile.ZipFile(zip_file, mode='w')
    zf.write(csv_file, os.path.basename(csv_file), compress_type=zipfile.ZIP_DEFLATED)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_make_small_masks()
    #run_make_small_images()

    print('\nsucess!')