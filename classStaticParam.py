import torch
import pymongo
import datetime


class classStaticParam:
    def __init__(self):
        # Device configuration
        # M1 不支援 cuda 所以只能用 CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """ DB setting """
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.coll_analyze_ticker = self.client['python_getStockNews']['analyze_ticker']
        self.coll_analyze_news = self.client['python_getStockNews']['analyze_news']
        self.coll_analyze_news_encoding = self.client['python_getStockNews']['analyze_news_encoding']
        self.coll_analyze_period = self.client['python_getStockNews']['analyze_period']

        """ recorder """
        self.training_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        """ local file path / name """
        self.external_ssd_path = '/Volumes/P2/School/Train_Data/'  # 外接硬碟
        self.sequence_combination_file_path = self.external_ssd_path + 'Combination_Sequence/'  # 組合資料（來源參考dissertation_03_DataPreprocessing）
        self.temp_path = 'TEMP_DATA/'
        self.model_path = 'Model/'
        self.model1_name = self.model_path + 'model1.pt'
        self.weights1_name = self.model_path + 'model1_weights.pt'
        self.model2_name = self.model_path + 'model2.pt'
        self.weights2_name = self.model_path + 'model2_weights.pt'
        self.log_path = 'LOG/' + self.training_datetime + '/'
        self.log_name = self.log_path + 'log_' + self.training_datetime + '.csv'

        """ train data """
        self.model1_train_feature_file_path = self.temp_path + 'model1_train_feature.csv'
        self.model1_train_label_file_path = self.temp_path + 'model1_train_label.csv'
        self.model2_train_feature_file_path = self.temp_path + 'model2_train_feature.csv'
        self.model2_train_label_file_path = self.temp_path + 'model2_train_label.csv'

        """ test data """
        self.model1_test_feature_file_path = self.temp_path + 'model1_test_feature.csv'
        self.model1_test_label_file_path = self.temp_path + 'model1_test_label.csv'
        self.model2_test_feature_file_path = self.temp_path + 'model2_test_feature.csv'
        self.model2_test_label_file_path = self.temp_path + 'model2_test_label.csv'

        """ model & log backup in external SSD """
        self.back_up_path = self.external_ssd_path + 'Model_Back_Up/model1 ' + self.training_datetime + '/'
        self.back_up_model_name = self.back_up_path + 'model1.pt'
        self.back_up_weights_name = self.back_up_path + 'model1_weights.pt'
        self.back_up_log_name = self.back_up_path + 'log.csv'
