import torch
from storyseed.corpus import ROOT
from storyseed.training_config import TrainingConfig,data_identity
from storyseed.training_data import TrainingBatches
from storyseed.training_evaluation import evaluate_decoder
from storyseed.decoder import DecoderLanguageModel
from storyseed.build_evidence import record


def main():
    config=TrainingConfig.load(ROOT/'configs/phase06_teaching.json')
    torch.set_num_threads(config.threads);torch.manual_seed(config.seed)
    train=TrainingBatches('train',64,32);dev=TrainingBatches('development',64,32)
    assert sum(train.target_counts)==359933 and sum(dev.target_counts)==39711
    model=DecoderLanguageModel(config.model).train()
    metrics=evaluate_decoder(model,dev,config.evaluation_batch_rows)
    assert model.training and metrics['prediction_positions']==39711
    rows=[0,1,len(train)-1];batch=train.batch(rows)
    record(6,'01_run_contract',{'command':'.venv/Scripts/python scripts/phase06/01_define_run.py',
       'configuration':config.to_dict(),'data_identity':data_identity(),'training_target_population':sum(train.target_counts),
       'development':metrics,'selected_rows':rows,'real_input_counts':batch['attention_mask'].sum(-1).tolist(),
       'scored_target_counts':batch['loss_mask'].sum(-1).tolist(),'evaluation_restored_training_mode':True,
       'context_policy':'up to context-score_tokens warm-up inputs before each disjoint scored block; window-relative positions',
       'final_holdout_accessed':False})

if __name__=='__main__':main()
