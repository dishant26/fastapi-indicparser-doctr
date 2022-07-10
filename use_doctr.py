import os
from uuid import uuid4
import doctr_dependency
try:
    from doctr.io import DocumentFile
    from doctr.models.zoo import ocr_predictor
except:
    print('DocTR not found. Installing. ')
    doctr_dependency.install_doctr()
try:
    from fastapi import FastAPI ,UploadFile
except:
    print('FastAPI not found. Installing')
    doctr_dependency.install_fastapi()

app = FastAPI()

class doctr:
    def __init__(self,file: UploadFile, pretrained=True):
        self.FILE = file
        self.pretrained = pretrained


    def ocr(self):
        """
        this method takes the file be it pdf or an image and after doing OCR it returns the label studio compatible json annotations.
        """

        ###################### O C R     B L O C K ##########################
        os.environ['USE_TORCH'] = '1'
        doc = self.FILE.file.read()
        if self.FILE.filename.endswith('pdf'):
            DOC = DocumentFile.from_pdf(doc)
        elif self.FILE.filename.endswith(('jpg', 'jpeg', 'png')):
            DOC = DocumentFile.from_images(doc)
        else:
            raise TypeError('The format of the document is not supported. Make sure it is pdf or jpg/jpeg/png.')
        predictor = ocr_predictor(pretrained=self.pretrained)
        output = predictor(DOC)
        model_output = output.export()
        ###################### C O N V E R T    T O   L A B E L  S T U D I O    F O R M A T     B L O C K ###########################

        img_height, img_width = DOC[0].shape[0], DOC[0].shape[1]

        file_name = self.FILE.filename #os.path.basename(self.path)
        URL = f'http://localhost:8081/{file_name}'

        results = []
        all_scores = []
        
        for block in model_output['pages'][0]['blocks']:
            for line in block['lines']:
                words = []
                confidences = []
                for word in line['words']:
                    words.append(word['value'])
                    confidences.append(word['confidence'])
                    width = word['geometry'][1][0] - word['geometry'][0][0]
                    height = word['geometry'][1][1] - word['geometry'][0][1]
                    bbox = {
                        'x': 100 * word['geometry'][0][0] / img_width,
                        'y': 100 * word['geometry'][0][1] / img_height,
                        'width': 100 * width / img_width,
                        'height': 100 * height / img_height,
                        'rotation': 0
                    }
                text = ' '.join(str(v) for v in words).strip()
                if not text:
                    continue
                region_id = str(uuid4())[:10]
                score = sum(confidences) / len(confidences) if confidences else 0
                bbox_result = {
                    'id': region_id, 'from_name': 'bbox', 'to_name': 'image',
                    'type': 'rectangle', 'value': bbox
                }
                transcription_result = {
                    'id': region_id, 'from_name': 'transcription', 'to_name': 'image',
                    'type': 'textarea', 'value': dict(text=[text], **bbox), 'score': score
                }
                results.extend([bbox_result, transcription_result])
                all_scores.append(score)

        return {
            'data': {
                'ocr': URL
            },
            'predictions': [{
                'result': results,
                'score': sum(all_scores) / len(all_scores) if all_scores else 0
            }]
        }
