# FastAPI for [Indic Parser](https://github.com/document-analysis-tools/indic-parser) & [docTR](https://github.com/mindee/doctr)

1. Clone the Respository
2. ```cd fastapi-indicparser-doctr```
3. Install all the packages ```pip install -r packages.txt```
4. Run app.py ```uvicorn app:app --reload```
5. Send <b>POST</b> request


## For OCR
```
curl -X 'POST' \
  'http://127.0.0.1:8000/indicparser' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@GK2_page-0280.jpg;type=image/jpeg' \
  -F 'inference=no' \
  -F 'lang=san_iitb'
```

## For Layout Detection
```
curl -X 'POST' \
  'http://127.0.0.1:8000/indicparser' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@GK2_page-0284.jpg;type=image/jpeg' \
  -F 'inference=yes' \
  -F 'lang=san_iitb' \
  -F 'model=Sanskrit_PubLayNet_faster_rcnn' \
  -F 'confidence_threshold=0.5'
```
## For docTR
```
curl -X 'POST' \
  'http://127.0.0.1:8000/doctr?model=DocTR' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'File=@sample.jpeg;type=image/jpeg'
```
