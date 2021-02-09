#include "detector.h"
#include "darknet.h"
#include "opencv2/opencv.hpp"
#include <pthread.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <arpa/inet.h>
#include <cstring>
#include <x264.h>

#include <vector>

///////////////////////////////////////yuv///////////////////////////////////////

#define IN_DECODE_TYPE X264_CSP_RGB
#define OUT_DECODE_TYPE X264_CSP_RGB

static bool encoder_can_use = false;

static x264_t * pHandle = NULL;

static x264_param_t * pParam = NULL;

x264_nal_t * pNals = NULL;

int i_nal = 0;

static int video_width,video_height;

/**
 *
 */
void clean_mem() {
    if(encoder_can_use) {
        encoder_can_use = NULL;
        if (pHandle != NULL) {
            x264_encoder_close(pHandle);
            pHandle = NULL;
        }
        if (pParam != NULL) {
            free(pParam);
            pParam = NULL;
        }
    }
}

#define h264_now_fps 10
void alloc_and_init_profile() {
    //alloc
    pParam = (x264_param_t *)malloc(sizeof(x264_param_t));
    //init
    int status = x264_param_default_preset(pParam,x264_preset_names[1],x264_tune_names[7]);
    if(status >= 0) {
        pParam->i_threads = X264_SYNC_LOOKAHEAD_AUTO;
        int screen_width = video_width;
        int screen_height = video_height;
        pParam->i_width = screen_width;
        pParam->i_height = screen_height;
        pParam->i_fps_num = h264_now_fps;
        pParam->i_fps_den = 1;
        pParam->i_keyint_max = h264_now_fps;
        pParam->b_intra_refresh = 1;
        pParam->rc.i_rc_method = X264_RC_CRF;
        pParam->rc.f_rf_constant = 25;
        pParam->rc.f_rf_constant_max = 45;
        pParam->b_repeat_headers = 1;
        pParam->b_annexb = 1;
        pParam->i_csp = OUT_DECODE_TYPE;
        pParam->i_bframe = 0;
        pParam->i_log_level = X264_LOG_INFO;
        status = x264_param_apply_profile(pParam, x264_profile_names[1]);
    }
}

void alloc_x264_t() {
    pHandle = x264_encoder_open(pParam);
}

void initTool(int width,int height) {
    video_width = width;
    video_height = height;
    alloc_and_init_profile();
    alloc_x264_t();
    i_nal = 0;
    pNals = NULL;
    encoder_can_use = true;
}

void initYUVTool(int width, int height) {
    clean_mem();
    initTool(width,height);
}

size_t encode_rgb_frame(void * ori_data,size_t ori_data_size,void * return_data_buffer, uint8_t return_data_buffer_size) {
    int return_size = 0;
    if(encoder_can_use && ori_data && ori_data_size > 0) {
        x264_picture_t pPic_in;
        x264_picture_t pPic_out;
        x264_picture_init(&pPic_out);
        //
        x264_picture_alloc(&pPic_in, IN_DECODE_TYPE, video_width, video_height);
        memcpy(pPic_in.img.plane[0],ori_data, ori_data_size);
        int i_frame_size = x264_encoder_encode(pHandle,&pNals,&i_nal,&pPic_in,&pPic_out);
        if(i_frame_size > 0) {
            return_size = i_frame_size;
            if(return_data_buffer_size > i_frame_size) {
                //write_data
                int offset = 0;
                for(int i = 0 ; i < i_nal ; i ++ ) {
                    uint8_t * p_payload = (pNals + i)->p_payload;
                    int i_payload = (pNals + i)->i_payload;
                    memcpy(((uint8_t *)return_data_buffer)+offset,p_payload,(size_t)i_payload);
                    offset += i_payload;
                }
            }
        }
        pNals = NULL;
        x264_picture_clean(&pPic_in);
    }
    return return_size;
}

void clean_env() {
    //__android_log_print(ANDROID_LOG_ERROR,"jni","Java_org_enes_lanvideocall_utils_video_X264Util_clean_1env");
    clean_mem();
}

//////////////////////////////start /////////////////////////////////




static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = (network**)calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class_ = j;
            if (dets[i].prob[class_]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class_],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = (FILE** )calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = (image*)calloc(nthreads, sizeof(image));
    image *val_resized = (image*)calloc(nthreads, sizeof(image));
    image *buf = (image*)calloc(nthreads, sizeof(image));
    image *buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = (FILE**)calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = (image*)calloc(nthreads, sizeof(image));
    image *val_resized = (image*)calloc(nthreads, sizeof(image));
    image *buf = (image*)calloc(nthreads, sizeof(image));
    image *buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    printf("%s\n",datacfg);
    printf("%s\n",cfgfile);
    printf("%s\n",weightfile);
    printf("%s\n",filename);
    printf("%f\n",thresh);
    printf("%f\n",hier_thresh);
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //printf("test : %d,%d\n ",sized.w, sized.h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];

        printf("net->n=%d\n",net->n);
        printf("l.classes=%d\n", l.classes);
        printf("l.type=%d\n", l.type);


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        // printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}


image my_ipl_to_image(IplImage* src) {
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

image my_mat_to_image(cv::Mat mat) {
    IplImage ipl = mat;
    image im = my_ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

IplImage *my_image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

cv::Mat my_image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = my_image_to_ipl(copy);
    cv::Mat m = cv::cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

//////////////////////////////////////// picture size /////////////////////////////////
#define kPictureWidth 640
#define kPictureHeight 512
#define kPictureChannels 3

/////////////////////////////////////// server ////////////////////////////////////////
#define kServerAddress "192.168.2.10"
#define kServerPort 30001

/////////////////////////////////////// mobile client /////////////////////////////////
#define kMobileClientAddress "192.168.2.12"
#define kMobileClientPort 30002

// void readFileToMem(void * buff, size_t size, std::string file_to_path) {
//     if(size > 0) {
//         std::cout << "read file from: " << file_to_path << std::endl;
//         FILE * fptr;
//         fptr = fopen(file_to_path.c_str(),"rb");
//         if(fptr == NULL) {
//             std::cout << "File Read Error" << std::endl;
//             exit(1);
//         }
//         fread(buff, size, 1, fptr);
//         fclose(fptr);
//     }
// }

// struct send_thread_struct {
//     std::vector<uint8_t> buff_jpg;
//     struct sockaddr_in mobile_udp_in;
//     int mobile_udp_sock_fd;

// } send_args;

// void * send_thread(void * arg) {
//     if(arg != NULL) {
//         struct send_thread_struct * arg_struct = (struct send_thread_struct *)arg;
//         std::vector<uint8_t> buff_jpg = arg_struct->buff_jpg;
//         struct sockaddr_in mobile_udp_in = arg_struct->mobile_udp_in;
//         int mobile_udp_sock_fd = arg_struct->mobile_udp_sock_fd;
        
//         unsigned int jpg_size = buff_jpg.size();


//         printf("jpeg size:::%d\n",jpg_size);

//         ssize_t sent_count = sendto(mobile_udp_sock_fd, buff_jpg.data(), jpg_size, 0, (struct sockaddr *)& mobile_udp_in, sizeof(mobile_udp_in));
//         printf("sent count %d\n",sent_count);

//     }
// }

struct detect_thread_struct {
    network * net;
    void * picture;
    size_t picture_size;
    float thresh;
    float hier_thresh;
    char **names;
    image **alphabet;
    struct sockaddr_in mobile_udp_in;
    int mobile_udp_sock_fd;
} args;

void * detect_thread(void * arg) {
    if(arg != NULL) {
        //
        detector_is_detect_thread_running = true;
        //
        struct detect_thread_struct * arg_struct = (struct detect_thread_struct *)arg;
        // network * net = arg_struct->net;
        void * picture = arg_struct->picture;
        size_t picture_size = arg_struct->picture_size;
        float thresh = arg_struct->thresh;
        float hier_thresh = arg_struct->hier_thresh;
        char **names = arg_struct->names;
        image **alphabet = arg_struct->alphabet; 
        struct sockaddr_in mobile_udp_in = arg_struct->mobile_udp_in;
        int mobile_udp_sock_fd = arg_struct->mobile_udp_sock_fd;
        //
        // layer l = net->layers[net->n-1];
        //
        cv::Mat mat(kPictureHeight, kPictureWidth, CV_8UC3, picture, kPictureWidth*3);
        // note :: this is the RGB COLOR
        // image im = my_mat_to_image(mat);
        //
        // image sized = letterbox_image(im, net->w, net->h);
        //
        // float *X = sized.data;
        //
        // double time = what_time_is_it_now();
        // network_predict(net, X);
        // std::cout << "Predicted in " << what_time_is_it_now() - time << " seconds." << std::endl;
        //
        // int nboxes = 0;
        // detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        // printf("%d\n",nboxes);
        // float nms=.60;
        // if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        //draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        //free_detections(dets, nboxes);
        //
        // free_image(sized);
        // mat.release();
        // cv::Mat m = my_image_to_mat(im);
        // convert to x264
        // void * data = m.data;
        // uint8_t buffer_size = sizeof(uint8_t) * picture_size;

        std::vector<uint8_t> buff_jpg;
        std::vector<int> jpg_param(2);
        jpg_param[0] = cv::IMWRITE_JPEG_QUALITY;
        jpg_param[1] = 60;
        cv::imencode(".jpg", mat, buff_jpg, jpg_param);
        unsigned int jpg_size = buff_jpg.size();


        // printf("jpeg size:::%d\n",jpg_size);
        // void * buffer = malloc(buffer_size);
        // int return_h264_frame_size = encode_rgb_frame(data,picture_size,buffer,buffer_size);
        // printf("return h264 frame size: %d \n", return_h264_frame_size);
        ssize_t sent_count = sendto(mobile_udp_sock_fd, buff_jpg.data(), jpg_size, 0, (struct sockaddr *)& mobile_udp_in, sizeof(mobile_udp_in));
        // int delay = 1000/10;
        printf("sent count %d\n",sent_count);

        // imshow("ttttt",m);
        // cv::waitKey(1);
        // imwrite("/tmp/aaa.jpg",m);
        // save_image(im, "/tmp/a.jpg");

        // free(buffer);
        //
        mat.release();

        // struct send_thread_struct ss;
        // ss.buff_jpg = buff_jpg;
        // ss.mobile_udp_in = mobile_udp_in;
        // ss.mobile_udp_sock_fd = mobile_udp_sock_fd;
        // pthread_create
        // pthread_t new_thread;
        // pthread_create(&new_thread, NULL, &send_thread, &ss);


        //              
        // free_image(im);
        //
        detector_is_detect_thread_running = false;
    }
}

void test_detector_v2(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    printf("Method: test_detector_v2 \n");
    printf("Try to loading x264 library\n");
    initYUVTool(kPictureWidth, kPictureHeight);
    printf("Try to loading cfg now \n");
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    printf("Try to loading network now \n");
    image **alphabet = load_alphabet();
    // network *net = load_network(cfgfile, weightfile, 0);
    //set_batch_network(net, 1);
    srand(2222222);

    // start loading tcp network
    printf("Try to connect with the server\n");

    int listening_port = kServerPort;
    const char * listening_address = kServerAddress;
    struct sockaddr_in clientSockAddr;
    int clientFd = socket(AF_INET, SOCK_STREAM, 0);
    
    if(clientFd < 0) {
        printf("can not create client socket fd \n");
        exit(1);
    }

    clientSockAddr.sin_family = AF_INET;
    clientSockAddr.sin_port = htons(listening_port);
    clientSockAddr.sin_addr.s_addr = inet_addr(listening_address);
    printf("tcp socket fd: %d \n", clientFd);
    size_t sockaddr_in_size = sizeof(struct sockaddr_in);

    // loading udp network
    printf("Try to initializing the UDP sender \n");
    struct sockaddr_in mobileSockAddr;
    int mobileSockFd = socket(AF_INET, SOCK_DGRAM, 0);

    if(mobileSockFd < 0) {
        printf("can not create mobile udp socket fd \n");
        exit(1);
    }

    printf("mobile udp socket fd: %d \n", mobileSockFd);

    mobileSockAddr.sin_family = AF_INET;
    mobileSockAddr.sin_port = htons(kMobileClientPort);
    mobileSockAddr.sin_addr.s_addr = inet_addr(kMobileClientAddress);


    size_t picture_size = kPictureWidth * kPictureHeight * kPictureChannels * sizeof(uint8_t);
   
    // set buffer size
    int buffer_size = 
    
    
    ;//409600;
    // allocate a buffer
    void * tmp_ptr = malloc(buffer_size);
    memset(tmp_ptr,0,buffer_size);
    //
    printf("full picture size : width:%d, height:%d, channels:%d \n",kPictureWidth,kPictureHeight,kPictureChannels);
     void * picture = malloc(picture_size);
    int now_file_count = 0;


    // bool is_first_packet_start = false;
    //
    while(true) {
        printf("try to connect with %s:%d\n",listening_address, listening_port);
        int connect_result = connect(clientFd, (struct sockaddr *)&clientSockAddr, sockaddr_in_size);
        printf("connect to %s:%d successful\n",listening_address, listening_port);
        while(connect_result == 0) {
            int read_count = recv(clientFd, tmp_ptr, buffer_size, 0);
            // printf("read data count %d.\n",read_count);
            if(read_count == 0) {
                printf("EOF !!!!! Try to reconnect.\n");
                connect_result = -1;
            } else if(read_count > 0){
                if(read_count != 2) {
                    if(now_file_count + read_count <= picture_size) {
                            memcpy((char *)picture+now_file_count,tmp_ptr,read_count);
                    }
                    now_file_count += read_count;
                } else {
                    now_file_count = 0;
                    // printf("full \n");



                    void * picture_to_detect_thread = malloc(picture_size);
                    memcpy(picture_to_detect_thread, picture, picture_size);

                    cv::Mat mat(kPictureHeight, kPictureWidth, CV_8UC3, picture_to_detect_thread, kPictureWidth*3);
                    std::vector<uint8_t> buff_jpg;
                            std::vector<int> jpg_param(2);
                            jpg_param[0] = cv::IMWRITE_JPEG_QUALITY;
                            jpg_param[1] = 50;
                            cv::imencode(".jpg", mat, buff_jpg, jpg_param);
                            unsigned int jpg_size = buff_jpg.size();
                            ssize_t sent_count = sendto(mobileSockFd, buff_jpg.data(), jpg_size, 0, (struct sockaddr *)& mobileSockAddr, sizeof(mobileSockAddr));
                            // // int delay = 1000/10;
                            // printf("sent count %d\n",sent_count);
                            mat.release();


                            free(picture_to_detect_thread);
                    

                }
            }

            

            // std::string part_str = (char *)tmp_ptr;
            
            // size_t is_find_end_str_pos = part_str.find("\r\n");
            





            // if(read_count == 0) {
            //     printf("EOF !!!!! Try to reconnect.\n");
            //     connect_result = -1;
            // } else if(read_count > 0){
            //     if(read_count == picture_size) {
            //         printf("Single File !!!!!\n");
            //     } else {

            //         if(now_file_count + read_count <= picture_size) {
            //                 memcpy((char *)picture+now_file_count,tmp_ptr,read_count);
            //         }
            //         now_file_count += read_count;

            //         if(now_file_count == picture_size) {
            //             now_file_count = 0;
            //             if(!detector_is_detect_thread_running) {
            //                 // drop it to new thread
            //                 pthread_t new_thread;
            //                 struct detect_thread_struct args;
            //                 args.net = net;
            //                 void * picture_to_detect_thread = malloc(picture_size);
            //                 memcpy(picture_to_detect_thread, picture, picture_size);
            //                 args.picture_size = picture_size;
            //                 args.picture = picture_to_detect_thread;
            //                 args.thresh = thresh;
            //                 args.hier_thresh = hier_thresh;
            //                 args.names = names;
            //                 args.alphabet = alphabet;
            //                 args.mobile_udp_in = mobileSockAddr;
            //                 args.mobile_udp_sock_fd = mobileSockFd;
            //                 // pthread_create
            //                 pthread_create(&new_thread, NULL, &detect_thread, &args);
            //             }
            //             // else {
            //             //     printf("frame droped \n");
            //             // }
            //         } else if(now_file_count > picture_size) {
            //             // printf("except frame , drop it.\n");
            //             now_file_count = 0;
            //         }
            //     }
            // }
        }
        printf("retry after 3sec.\n");
        sleep(3);
    }
    free(picture);
    free(tmp_ptr);
    // free(net);
    clean_env();
    close(clientFd);
    close(mobileSockFd);
}



/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }


    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }main
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
