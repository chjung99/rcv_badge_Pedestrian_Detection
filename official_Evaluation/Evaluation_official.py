### Evaluation
import os
import matplotlib.pyplot as plt
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval

annType = 'bbox'
JSON_GT_FILE = os.path.join( './kaist_annotations_test20.json' )
cocoGt = COCO(JSON_GT_FILE)

def evaluate_coco(test_json_path):

    fig_test,  ax_test  = plt.subplots(figsize=(18,15))
         
    rstFile = os.path.join(test_json_path)

    try:
        cocoDt = cocoGt.loadRes(rstFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = [1]    
        cocoEval.evaluate(0)
        cocoEval.accumulate()
        curPerf = cocoEval.summarize(0)    
        cocoEval.draw_figure(ax_test, rstFile.replace('json', 'jpg'))        
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

if __name__ == '__main__':
    #2020_April_Pedestrian_Detection_Challenge/2019.RCVSS/HandOnLabs_Detection/Sejong&FLIR/jobs/2020-04-03_04h44m_/COCO_TEST_det_checkpoint_ssd300.pth.tar090.json
    test_json_path = './COCO_TEST_det_checkpoint_ssd300.pth.tar008.json'
    evaluate_coco(test_json_path)
