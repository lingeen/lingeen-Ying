# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 1:49 PM
# @Author  : Kevin

project_home_path_prefix="/Users/kevin/Downloads/work/pycharm/Ying"

user_dict_path=project_home_path_prefix+"/data/keywords.txt"
stop_words_path=project_home_path_prefix+"/data/stopwords.txt"



###########意图识别################
think_train_data_path=project_home_path_prefix+"/data/think/data_by_word_train.txt"
think_train_test_path=project_home_path_prefix+"/data/think/data_by_word_test.txt"
train_data_batch_size=200
think_intention_recognition_model_path=project_home_path_prefix+"/model/think_intention_recognition_model.model"


#################################

#############召回模型#############

recall_question_cut_by_character_path=project_home_path_prefix+"/data/search/question_cut_by_character.txt"
recall_merged_q_path=project_home_path_prefix+"/data/search/question_cut_by_character.txt"
recall_fasttext_vector_model_path=project_home_path_prefix+"/model/recall_fasttext_vector_model.model"
recall_pysparnn_cp_model_path=project_home_path_prefix+"/model/recall_pysparnn_cp_model.model"
#################################

#############排序模型#############
sort_ws_model_path=project_home_path_prefix+"/model/sort_word_sequence_model.model"
sort_all_file_path=project_home_path_prefix+"/data/sort/questions_all.txt"
sort_data_batch_size=100
sort_input1_file_path=project_home_path_prefix+"/data/sort/questions_input1.txt"
sort_input2_file_path=project_home_path_prefix+"/data/sort/questions_input2.txt"
sort_label_file_path=project_home_path_prefix+"/data/sort/input1_input2_label.txt"

sort_input_embedding_dim=200
sort_gru1_hidden_size=300
sort_gru1_layers=2
sort_gru2_hidden_size=200
sort_gru2_layers=1
dnn_linear1_out_dim=256
# 二分类
dnn_label_size=2
sort_drop_out=0.5
#################################

#############闲聊模型#############
chat_qingyun_file_path=project_home_path_prefix+"/data/chat/qingyun.tsv"
chat_ask_file_path=project_home_path_prefix+"/data/chat/ask_cut_by_character_clean.txt"
chat_answer_file_path=project_home_path_prefix+"/data/chat/answer_cut_by_character_clean.txt"
chat_data_batch_size=64

chat_word_sequence_model_path=project_home_path_prefix+"/model/chat_word_sequence_model.model"
chat_ask_word_sequence_model_path=project_home_path_prefix+"/model/chat_ask_word_sequence_model.model"
chat_answer_word_sequence_model_path=project_home_path_prefix+"/model/chat_answer_word_sequence_model.model"
caht_train_grad_clip_max=0.01
chat_encoder_embedding_dim=200
chat_gru_num_layers=1
chat_gru_bidirectional=False
chat_gru_dropout=0.4

chat_decoder_embedding_dim=200

chat_decoder_beam_width=3

chat_ask_sentence_max_len=10
chat_answer_sentence_max_len=10

chat_encoder_gru_hidden_size=300
chat_decoder_gru_hidden_size=300

chat_decoder_attention_match_type="concat"#["dot","general","concat"]

chat_teacher_forcing_ratio=0.5

chat_seq_2_seq_model_path=project_home_path_prefix+"/model/chat_seq_2_seq_model.pkl"
chat_seq_optimizer_model_path=project_home_path_prefix+"/model/chat_seq_optimizer_model.pkl"
#################################
