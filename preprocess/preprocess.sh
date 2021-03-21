#!/bin/bash -x

pip install -r requirements.txt
## python3-mecab用のunidic
python -m unidic download
##cuda10.1用のtorch
pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
##


DATA_DIR="../data/"
if [ ! -d ${DATA_DIR} ]; then
  mkdir ${DATA_DIR}
fi

wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json -P ${DATA_DIR}
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json -P ${DATA_DIR}
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json -P ${DATA_DIR}
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz -P ${DATA_DIR}
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/all_entities.json.gz -P ${DATA_DIR}

python build_inverted_index.py
python get_contexts.py
python make_inputs.py