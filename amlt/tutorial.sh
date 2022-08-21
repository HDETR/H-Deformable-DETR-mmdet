pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/7e404de797f4e1eeca406c1739b00867
amlt project checkout --create decoder_aug_v2 openseg # 9fhoDNYPRFs8zc53B13GDqvEVynmOEe+RchBP8p5d6EWsU04LVfjs3+Q4ab7r2Pq9EXqylgYH28gVmLDk95hqQ==

amlt run A100.yaml  decoder_aug_exp_batch_x6

amlt run V100.yaml  h_deformable_detr_varify_exps