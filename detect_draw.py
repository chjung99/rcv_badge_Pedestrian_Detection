from torchcv.datasets.transforms import *
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import os.path
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model checkpoint
#checkpoint = 'jobs/2020-04-15_06h58m_sch_changed2/checkpoint_ssd300.pth.tar024'
checkpoint = 'jobs/2020-05-12_15h56m_test/checkpoint_ssd300.pth.tar031'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()
#Data load
#DB_ROOT = './Sejong_RCV_datasets/Images/KAIST/set08/V002/visible/'
DB_ROOT='./kaist-rgbt/images/set07/V002/visible'
# {SET_ID}/{VID_ID}/{MODALITY}/{IMG_ID}.jpg
lwir_image_list = os.listdir(os.path.join(DB_ROOT))
#import pdb;pdb.set_trace()
lwir_image_list.sort()
ids = list()
for ii, image_lwir in enumerate(lwir_image_list):
    ids.append(image_lwir)

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD_512x640, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    # Transform
    Transform = Compose([Resize([512,640]),\
                    ToTensor(),\
                    Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R') #8bit
                        ]) #8bit  
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    image = Transform(original_image)
    
    #import pdb;pdb.set_trace()
    # Move to default device
    image = image[0].to(device)
    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)  
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    # Transform to original image dimensions
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    import pdb;pdb.set_trace()
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD_512x640.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
            return original_image
    # Annotate
    annotated_image = lwir
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])  
        # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness
        # Text
        text_score = str(det_scores[0][i])[:7]
        text_size = font.getsize(text_score)
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
        draw.text(xy=text_location, text='{:.4f}'.format(det_scores[0][i].item()), fill='white', font=font)
    # del draw
    return annotated_image
if __name__ == '__main__':
    for ii in range(len(ids)):
        frame_id = ids[ii]
        lwir_file_name = frame_id
        #import pdb;pdb.set_trace()
        lwir = Image.open( os.path.join(DB_ROOT,lwir_file_name) )
        
        #import pdb;pdb.set_trace()
        annotate_lwir = detect(lwir, min_score=0.1, max_overlap=0.45, top_k=200)
        #annotate_lwir.save('./test/I{:06d}.jpeg'.format(ii))
        annotate_lwir.save('./test/I{:06d}.jpeg'.format(ii))
        print('%d.jpeg saved' %ii)