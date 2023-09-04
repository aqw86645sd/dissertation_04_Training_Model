from classModel1 import LSTMModel1
from classModel2 import KMEANSModel2
from LineNotifyMessage import line_notify_message
from classStaticParam import classStaticParam  # 固定參數設定檔

import gc
import os
import csv
import glob
import json
import math
import time
import numpy
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


class Entrance(classStaticParam):
    def __init__(self):

        super().__init__()

        self.remark = 'move to new repository'

        """ train data """
        self.p_ticker = 'AAPL'
        self.p_start_date = '2022-01-01'
        self.p_end_date = '2022-06-30'

        """ model1 訓練資料設定 """
        self.model1_training_data_per_day = 3000  # model1 每個交易日要訓練多少筆資料
        self.model1_separate_file_number = 300  # model1 將每日訓練資料分成多少檔案，方便step2運算 （大於1的整數，且要可以整除training_data_per_day）
        self.model1_training_data_cnt_per_file = self.model1_training_data_per_day // self.model1_separate_file_number  # 每個檔案有多少筆資料

        self.train_test_ratio = 0.8  # 訓練資料與測試資料佔比

        """ model parameter """
        self.p_encoding_length = 12  # encoding句子長度
        self.p_batch_size = 3
        self.p_vocab_size = 1391  # 字典數量
        self.p_embedding_dim = 64
        self.p_hidden_dim = 80
        self.p_dropout = 0.2
        self.p_epochs = 100  # 訓練次數
        self.p_learning_rate = 3e-4  # 優化器學習率

        """ 是否要備份 """
        self.p_backup_flag = False

    def run(self):
        """ 選擇要執行的 model 訓練 """
        self.run_model1()
        # self.run_model2()

    def run_model1(self):
        print('training model1 start')

        """ create temp folder and shuffle original combination """
        self.run_model1_step1()

        """ encoding """
        self.run_model1_step2()

        """ model1 training """
        self.run_model1_step3()

        """ model1 predict """
        self.run_model1_step4()

        print('training model1 end')

    def run_model2(self):
        print('training model2 start')

        """ create model2 dataset """
        self.run_model2_step1()

        """ model2 training """
        # self.run_model2_step2()

        """ model2 predict """

        """ model2 use news predict """

        print('training model2 end')

    def run_model1_step1(self):
        """
            create temp folder and shuffle original combination
        """
        print('model1 step1 start')
        start_time = time.time()

        """ check folder """
        # 確認組合檔案路徑是否存在
        if not os.path.exists(self.sequence_combination_file_path):
            print(self.sequence_combination_file_path, '路徑不存在！！！！')
            return

        """ remove temp folder and recreate """
        if os.path.exists(self.temp_path):
            shutil.rmtree(self.temp_path)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        """ get date list """
        training_date = []  # 要訓練的日期
        # 計算總天數＊幾輪次數，當作queue的最長值
        ticker_data = self.coll_analyze_ticker.find_one({'ticker': self.p_ticker})
        date_list = list(ticker_data['stock_data'])
        for p_date in date_list:
            if self.p_start_date <= p_date <= self.p_end_date:
                training_date.append(p_date)  # 要訓練的日期

        # 將日期隨機
        random.shuffle(training_date)

        for p_date in training_date:

            print('Next date:', p_date)

            # 原始組合資料檔案
            combination_file_name = self.sequence_combination_file_path + self.p_ticker + '_' + p_date + '.txt'
            with open(combination_file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # 讀取所有行
                random.shuffle(lines)  # 將順序打亂
                date_shuffle_combination_list = lines[0: self.model1_training_data_per_day]  # 將要計算的
                del lines  # clean parameter
                gc.collect()  # 清除或釋放未引用的記憶體

            for i in range(self.model1_separate_file_number):
                # 篩選後組合資料檔案 先存在 TEMP
                step1_file_name = 'shuffle_R' + str(i + 1) + '_' + self.p_ticker + '_' + p_date + '.txt'
                temp_step1_file_name = 'temp_shuffle_R' + str(i + 1) + '_' + self.p_ticker + '_' + p_date + '.txt'

                # 存放尚未完成的資料
                temp_step1_file_path = self.temp_path + temp_step1_file_name

                r_data_start_index = i * self.model1_training_data_cnt_per_file
                r_data_end_index = (i * self.model1_training_data_cnt_per_file) + self.model1_training_data_cnt_per_file
                r_data = date_shuffle_combination_list[r_data_start_index:r_data_end_index]

                # 每 10 筆存一次資料
                insert_data_str = ''
                for idx, data in enumerate(r_data):
                    insert_data_str += data
                    if (idx + 1) % 10 == 0 or (idx + 1) == self.model1_training_data_cnt_per_file:
                        with open(temp_step1_file_path, 'a') as t:
                            t.write(insert_data_str)

                            """ release memory """
                            del insert_data_str  # clean parameter
                            gc.collect()  # 清除或釋放未引用的記憶體
                            insert_data_str = ''  # default

                """ 將檔案從 temp_shuffle_file_name 轉移到 shuffle_file_name """
                step1_file_path = self.temp_path + step1_file_name
                os.rename(temp_step1_file_path, step1_file_path)  # rename
                print(step1_file_name)

                """ release memory """
                del step1_file_name  # clean parameter
                del step1_file_path  # clean parameter
                del temp_step1_file_name  # clean parameter
                del temp_step1_file_path  # clean parameter
                gc.collect()  # 清除或釋放未引用的記憶體

            """ release memory """
            del combination_file_name  # clean parameter
            del date_shuffle_combination_list  # clean parameter
            del r_data  # clean parameter
            gc.collect()  # 清除或釋放未引用的記憶體

        # 通知
        remark_msg = "\n備註: " + self.remark
        param_msg = "\n開始日期: {}\n結束日期: {}\n天數: {}\n每日訓練筆數: {}".format(self.p_start_date,
                                                                                      self.p_end_date,
                                                                                      str(len(training_date)),
                                                                                      str(self.model1_training_data_per_day))
        time_msg = self.function_calc_spend_time(start_time)
        line_notify_message('dissertation_04_Training_Model model1 step1 finished!' + remark_msg + param_msg + time_msg)

    def run_model1_step2(self):
        """
            使用預先處理好的組合產出 feature & label 的檔案
        """
        print('model1 step2 start')
        start_time = time.time()

        """ check folder """
        # 確認temp是否存在
        if not os.path.exists(self.temp_path):
            print(self.temp_path, '路徑不存在！！！！')
            return

        """ 找出 step1 產出的檔案 """
        step1_file_list = glob.glob(self.temp_path + "shuffle_R*.txt")

        for queue_step1_file_path in step1_file_list:

            with open(queue_step1_file_path, 'r', encoding='utf-8') as f:
                shuffle_combination_list = f.readlines()  # 讀取所有行

            for idx, date_combination_data in enumerate(shuffle_combination_list):
                combinations_data_json = json.loads(date_combination_data[:-1])  # 去掉斷行字元

                # label
                label = combinations_data_json['label']

                # sequence
                sequence_combinations = combinations_data_json['sequence_combinations']

                news_encoding_list = []  # encoding 後的 list

                # 找到該新聞encoding資料
                for news_sequence in sequence_combinations:
                    news_key = {'sequence': news_sequence}
                    analyze_news_encoding_data = self.coll_analyze_news_encoding.find_one(news_key)
                    if analyze_news_encoding_data is None:
                        # 無 encoding 資料
                        pass
                    else:
                        news_encoding_list.extend(analyze_news_encoding_data['news_encoding_list'])

                if len(news_encoding_list) == 0:
                    # 因為沒有字典裡的字所以長度為0，不需塞到資料集
                    continue

                # padding:預處理，將input統一長度
                if len(news_encoding_list) > self.p_encoding_length:
                    news_encoding_list = news_encoding_list[:self.p_encoding_length]
                else:
                    news_encoding_list.extend(
                        list([0 for _ in range(self.p_encoding_length - len(news_encoding_list))])
                    )

                if idx < self.model1_training_data_cnt_per_file * self.train_test_ratio:
                    """ train data """
                    feature_path = self.model1_train_feature_file_path
                    label_path = self.model1_train_label_file_path
                else:
                    """ test data """
                    feature_path = self.model1_test_feature_file_path
                    label_path = self.model1_test_label_file_path

                """ 將FEATURE存起來 """
                with open(feature_path, 'a') as f:
                    write = csv.writer(f)
                    write.writerow(news_encoding_list)

                """ 將LABEL存起來 """
                with open(label_path, 'a') as f:
                    write = csv.writer(f)
                    write.writerow([label])

                """ release memory """
                del news_encoding_list  # clean parameter
                gc.collect()  # 清除或釋放未引用的記憶體

            """ 已執行的檔案就先刪掉 """
            os.remove(queue_step1_file_path)  # remove finished file

            """ release memory """
            del queue_step1_file_path  # clean parameter
            del shuffle_combination_list
            del combinations_data_json  # clean parameter
            gc.collect()  # 清除或釋放未引用的記憶體

        """ release memory """
        del step1_file_list  # clean parameter
        gc.collect()  # 清除或釋放未引用的記憶體

        # 通知
        remark_msg = "\n備註: " + self.remark
        time_msg = self.function_calc_spend_time(start_time)
        line_notify_message('dissertation_04_Training_Model model1 step2 finished!' + remark_msg + time_msg)

    def run_model1_step3(self):
        """
            訓練 model1 模組
        """
        print('model1 step3 start')
        start_time = time.time()

        """ check folder """
        if not os.path.exists(self.model_path):
            # 建立 MODEL 主目錄
            os.makedirs(self.model_path)

        # 取得 feature
        with open(self.model1_train_feature_file_path) as f:
            reader = csv.reader(f)
            features = [row for row in reader]

            """ release memory """
            del reader  # clean parameter
            gc.collect()  # 清除或釋放未引用的記憶體

        # 取得 label
        with open(self.model1_train_label_file_path) as f:
            reader = csv.reader(f)
            labels = list(reader)

            """ release memory """
            del reader  # clean parameter
            gc.collect()  # 清除或釋放未引用的記憶體

        # CSV取得都是字串，要給numpy.array用就得改成int
        features = [[int(j) for j in i] for i in features]
        labels = [int(i[0]) for i in labels]

        """ features & labels """
        train_x = numpy.array(features)
        train_y = numpy.array(labels)

        train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=self.p_batch_size, drop_last=True)

        """ Model1 """
        if os.path.exists(self.model1_name):
            print('模組1已建立，匯入模型')
            # 從檔案中載入模型
            model1 = torch.load(self.model1_name)
            if os.path.exists(self.weights1_name):
                # 從檔案載入參數（不含模型）
                model1.load_state_dict(torch.load(self.weights1_name))
            model1.eval()
        else:
            print('無舊資料，建立新模組1')
            # 建構LSTM物件
            model1 = LSTMModel1(self.p_vocab_size, self.p_embedding_dim, self.p_hidden_dim, self.p_dropout)
            model1 = model1.to(self.device)

        # 優化器
        optimizer = torch.optim.Adam(model1.parameters(), lr=self.p_learning_rate)

        # 宣告損失函數
        criterion = nn.CrossEntropyLoss()

        print(model1)

        losses = []

        for e in range(self.p_epochs):

            h0, c0 = model1.init_hidden(self.p_batch_size, self.p_hidden_dim)

            h0 = h0.to(self.device)
            c0 = c0.to(self.device)

            for batch_idx, batch in enumerate(train_dl):
                input_ = batch[0].to(self.device)
                target_ = batch[1].to(self.device)

                optimizer.zero_grad()  # 模型的參數梯度初始化為0
                with torch.set_grad_enabled(True):
                    out, hidden = model1(input_, (h0, c0))
                    loss = criterion(out, target_)
                    loss.backward()
                    optimizer.step()  # 更新所有參數

            losses.append(loss.item())
            print(e + 1, loss.item())

        """ save model & weight """
        torch.save(model1, self.model1_name)
        torch.save(model1.state_dict(), self.weights1_name)

        """ check folder """
        if not os.path.exists(self.log_path):
            # 建立 LOG 主目錄
            os.makedirs(self.log_path)
        if not os.path.exists(self.back_up_path) and self.p_backup_flag:
            # 建立 back up 目錄
            os.makedirs(self.back_up_path)

        """ 將PARAMETER寫進LOG PATH """
        parameter_file_name = self.log_path + 'parameter.json'
        parameter_json = {
            'model': 'model1',
            'remark': self.remark,
            'ticker': self.p_ticker,
            'start_date': self.p_start_date,
            'end_date': self.p_end_date,
            'training_data_per_day': self.model1_training_data_per_day,
            'hidden_dim': self.p_hidden_dim,
            'vocab_size': self.p_vocab_size,
            'batch_size': self.p_batch_size,
            'epochs': self.p_epochs,
            'learning_rate': self.p_learning_rate
        }
        with open(parameter_file_name, 'a') as p:
            json.dump(parameter_json, p)

        # save history
        with open(self.log_name, 'a') as f:
            """ first row is title """
            log_title = ['index', 'loss']
            write = csv.writer(f)
            write.writerow(log_title)

            """ history data """
            for i in range(len(losses)):
                row = [i, losses[i]]
                write = csv.writer(f)
                write.writerow(row)

        """ backup """
        if self.p_backup_flag:
            shutil.copyfile(self.model1_name, self.back_up_model_name)
            shutil.copyfile(self.weights1_name, self.back_up_weights_name)
            shutil.copyfile(self.log_name, self.back_up_log_name)
            shutil.copyfile(self.log_path + 'parameter.json', self.back_up_path + 'parameter.json')

        # 通知
        remark_msg = "\n備註: " + self.remark
        param_msg = "\n訓練次數: {}".format(str(self.p_epochs))
        time_msg = self.function_calc_spend_time(start_time)
        line_notify_message('dissertation_04_Training_Model model1 step3 finished!' + remark_msg + param_msg + time_msg)

    def run_model1_step4(self):
        """
            預測並驗證 model1
        """
        print('model1 step4 start')
        start_time = time.time()

        """ check model file """
        if not os.path.exists(self.model1_name):
            print('沒有 model1')
            quit()

        # 從檔案中載入模型
        model1 = torch.load(self.model1_name)
        if os.path.exists(self.weights1_name):
            # 從檔案載入參數（不含模型）
            model1.load_state_dict(torch.load(self.weights1_name))
        model1.eval()

        # 優化器
        optimizer = torch.optim.Adam(model1.parameters(), lr=self.p_learning_rate)

        # 取得 feature
        with open(self.model1_test_feature_file_path) as f:
            reader = csv.reader(f)
            features = [row for row in reader]

            """ release memory """
            del reader  # clean parameter
            gc.collect()  # 清除或釋放未引用的記憶體

        # 取得 label
        with open(self.model1_test_label_file_path) as f:
            reader = csv.reader(f)
            labels = list(reader)

            """ release memory """
            del reader  # clean parameter
            gc.collect()  # 清除或釋放未引用的記憶體

        # CSV取得都是字串，要給numpy.array用就得改成int
        features = [[int(j) for j in i] for i in features]
        labels = [int(i[0]) for i in labels]

        """ features & labels """
        test_x = numpy.array(features)
        test_y = numpy.array(labels)

        test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        test_dl = DataLoader(test_ds, shuffle=True, batch_size=self.p_batch_size, drop_last=True)

        h0, c0 = model1.init_hidden(self.p_batch_size, self.p_hidden_dim)

        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        batch_acc = []
        for batch_idx, batch in enumerate(test_dl):
            input_ = batch[0].to(self.device)
            target_ = batch[1].to(self.device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out, hidden = model1(input_, (h0, c0))
                _, preds = torch.max(out, 1)
                preds = preds.to("cpu").tolist()
                batch_acc.append(accuracy_score(preds, target_.tolist()))

        accuracy = sum(batch_acc) / len(batch_acc)

        print('accuracy:', accuracy)

        # 通知
        remark_msg = "\n備註: " + self.remark
        acc_msg = "\naccuracy: " + str(accuracy)
        time_msg = self.function_calc_spend_time(start_time)
        line_notify_message('dissertation_04_Training_Model model1 step4 finished!' + remark_msg + acc_msg + time_msg)

    def run_model2_step1(self):
        """
            建立 model2 訓練用資料

            features : 『過去與當下行情資料』加上『當下的新聞經過model1的預測分類值』

                ＊ 行情資料：過去與當下行情資料去計算百分比後在分類到對應的區間 （同一個ticker在同一天的行情features會相同）
                ＊ model1的預測分類值：使用model1將該天新聞資料做好分類（features會依照news_id的新聞內容不一樣而不同）
                    作法：
                    1. 將相同news_id的資料抓出來
                    2. 將sequence排序（由小到大）
                    3. 將sequence的list，從index 0開始取三順位數然後加1在取三順位數，如0,1,2>1,2,3>2,3,4（取三個原因來自於當初新聞句子組合是取三句話）
                    4. 將同一個news_id做成model1的predict dataset，用model1對每筆dataset做預測，然後算平均預測值
                    5. 該news_id的label算法為：if 平均預測值 > 0.5 then 1 else 0

            labels : 使用未來的行情漲跌百分比區間當作label的依據
        """
        print('model2 step1 start')
        start_time = time.time()

        """ check model file """
        if not os.path.exists(self.model1_name):
            print('沒有 model1 無法預測')
            quit()

        # 從檔案中載入模型
        model1 = torch.load(self.model1_name)
        if os.path.exists(self.weights1_name):
            # 從檔案載入參數（不含模型）
            model1.load_state_dict(torch.load(self.weights1_name))
        model1.eval()

        # 優化器
        optimizer = torch.optim.Adam(model1.parameters(), lr=self.p_learning_rate)

        """ remove temp folder and recreate """
        if os.path.exists(self.temp_path):
            shutil.rmtree(self.temp_path)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        """ get date list """
        training_date = []  # 要訓練的日期
        ticker_data = self.coll_analyze_ticker.find_one({'ticker': self.p_ticker})
        date_list = list(ticker_data['stock_data'])
        for p_date in date_list:
            if self.p_start_date <= p_date <= self.p_end_date:
                training_date.append(p_date)  # 要訓練的日期

        # 將日期隨機
        random.shuffle(training_date)

        # 取得 ticker 行情
        ticker_key = {'ticker': self.p_ticker}
        db_ticker_json = self.coll_analyze_ticker.find_one(ticker_key)
        ticker_stock_data = db_ticker_json['stock_data']
        ticker_stock_date_list = list(ticker_stock_data)  # 用來抓 feature index

        # 資料架構為 (ticker 包含 (date 包含 news_id))

        for p_date in training_date:
            print('Next date:', p_date)

            """ 抓出行情的 features 部分 ＆ 宣告 label """
            # 該 ticker 行情在每個固定日期下都是固定的
            model2_features_fixed = None  # features固定資料
            model2_label = None
            if p_date in ticker_stock_date_list:
                date_index = ticker_stock_date_list.index(p_date)
                p_past_5_date = ticker_stock_date_list[date_index - 5]  # 抓出該日期過去五天營業日資料為feature依據
                p_past_20_date = ticker_stock_date_list[date_index - 20]  # 抓出該日期過去二十天營業日資料為feature依據
                p_future_date = ticker_stock_date_list[date_index + 5]  # 抓出該日期加五天營業日資料為label依據

                # 過去資訊
                past_price_5 = ticker_stock_data[p_past_5_date]['ticker_price']
                past_price_20 = ticker_stock_data[p_past_20_date]['ticker_price']

                # 當日資訊
                current_price = ticker_stock_data[p_date]['ticker_price']

                # 未來資訊
                future_price = ticker_stock_data[p_future_date]['ticker_price']

                # 計算過去行情與現在行情的漲幅百分比
                past_price_5_percent = (current_price - past_price_5) / past_price_5 * 100 / 2
                past_price_5_deviation = self.set_label_value(past_price_5_percent)

                past_price_20_percent = (current_price - past_price_20) / past_price_20 * 100 / 2
                past_price_20_deviation = self.set_label_value(past_price_20_percent)

                # 計算過去行情相關資料當作features
                model2_features_fixed = [past_price_5_deviation, past_price_20_deviation]

                # 計算未來時間行情與現在行情的漲幅百分比
                future_price_percent = (future_price - current_price) / current_price * 100
                model2_label = self.set_label_value(future_price_percent)

            """ find distinct ticker, news_id in a day """
            news_key = {'ticker': self.p_ticker, 'date': p_date, 'source': 'Zacks'}
            db_news_data = self.coll_analyze_news.find(news_key)

            # 取得新聞資料
            news_id_json = {}  # {news_id: {sequence: []}}

            for news_data in db_news_data:
                p_news_id = news_data['news_id']
                if p_news_id in news_id_json:
                    news_id_json[p_news_id]['sequence'].append(news_data['sequence'])
                else:
                    news_id_json[p_news_id] = {'sequence': [news_data['sequence']]}

            # 已經取得該ticker與該日期有哪些{news_id:{sequence:[]}}
            for p_news_id in news_id_json:
                # 排序
                sequence_list = news_id_json[p_news_id]['sequence']
                sequence_list.sort()

                news_encoding_list = []

                # 抓出該 news_id 下的 sequence 對應的 sentence encoding
                for p_sequence in sequence_list:
                    coll_key = {'sequence': p_sequence}
                    db_encoding = self.coll_analyze_news_encoding.find_one(coll_key)

                    if db_encoding:
                        news_encoding_list.append(db_encoding['news_encoding_list'])
                    else:
                        news_encoding_list.append([])

                predict_result_list = []  # 每三句話encoding的預測值

                # 讀取三句話
                for i in range(len(news_encoding_list) - 2):
                    ''' create train dataset '''
                    encoding_list = news_encoding_list[i]
                    encoding_list.extend(news_encoding_list[i + 1])
                    encoding_list.extend(news_encoding_list[i + 2])

                    """ padding """
                    if len(encoding_list) == 0:
                        continue
                    elif len(encoding_list) > self.p_encoding_length:
                        encoding_list = encoding_list[:self.p_encoding_length]
                    else:
                        encoding_list.extend(list([0 for _ in range(self.p_encoding_length - len(encoding_list))]))

                    """ 預測 model1 classification """
                    test_x = numpy.array([encoding_list])

                    test_ds = TensorDataset(torch.from_numpy(test_x))

                    test_dl = DataLoader(test_ds, shuffle=True, batch_size=1, drop_last=True)

                    h0, c0 = model1.init_hidden(batch_size=1, hidden_dim=self.p_hidden_dim)

                    h0 = h0.to(self.device)
                    c0 = c0.to(self.device)

                    for batch_idx, batch in enumerate(test_dl):
                        input_ = batch[0].to(self.device)

                        optimizer.zero_grad()
                        with torch.set_grad_enabled(False):
                            out, hidden = model1(input_, (h0, c0))
                            _, preds = torch.max(out, 1)
                            preds = preds.to("cpu").tolist()[0]
                            predict_result_list.append(preds)

                if len(predict_result_list) > 0:
                    # 該新聞有encoding資料才有label值
                    news_classification_label = 0 if sum(predict_result_list) / len(predict_result_list) < 0.5 else 1

                    model2_features = model2_features_fixed.copy()
                    model2_features.append(news_classification_label)

                    # 做成文件
                    """ train data """
                    feature_path = self.model2_train_feature_file_path
                    label_path = self.model2_train_label_file_path

                    """ 將FEATURE存起來 """
                    with open(feature_path, 'a') as f:
                        write = csv.writer(f)
                        write.writerow(model2_features)

                    """ 將LABEL存起來 """
                    with open(label_path, 'a') as f:
                        write = csv.writer(f)
                        write.writerow([model2_label])

        # 通知
        date_cnt_str = str(len(training_date))
        remark_msg = "\n備註: " + self.remark
        param_msg = "\n開始日期: {}\n結束日期: {}\n天數: {}".format(self.p_start_date, self.p_end_date, date_cnt_str)
        time_msg = self.function_calc_spend_time(start_time)
        line_notify_message('dissertation_04_Training_Model model2 step1 finished!' + remark_msg + param_msg + time_msg)

    def run_model2_step2(self):
        n_data = torch.ones(100, 2)
        xy0 = torch.normal(2 * n_data, 1)  # 生成均值为2，2 标准差为1的随机数组成的矩阵 shape=(100, 2)
        c0 = torch.zeros(100)
        xy1 = torch.normal(-2 * n_data, 1)  # 生成均值为-2，-2 标准差为1的随机数组成的矩阵 shape=(100, 2)
        c1 = torch.ones(100)
        X = torch.cat((xy0, xy1), 0)
        c = torch.cat((c0, c1), 0)
        plt.scatter(X[:, 0], X[:, 1], c=c, s=100, cmap='RdYlGn')
        plt.show()
        X.shape

        device = torch.device('cpu')
        k = KMEANSModel2(n_clusters=2, max_iter=10, verbose=False, device=device)
        y_pred = k.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=20)
        plt.scatter(k.centers[:, 0], k.centers[:, 1], c='red', s=80, alpha=.8)
        # representative_samples = x
        plt.scatter(X[k.representative_samples][:, 0], X[k.representative_samples][:, 1], c='blue', s=80, alpha=.8)
        plt.show()
        # k.centers, X[k.representative_samples] , metrics.calinski_harabasz_score(X, y_pred)

    # def run_model2_step2(self):
    #     """
    #         訓練 model2
    #     """
    #
    #     print('model2 step2 start')
    #     start_time = time.time()
    #
    #     """ check folder """
    #     if not os.path.exists(self.model_path):
    #         # 建立 MODEL 主目錄
    #         os.makedirs(self.model_path)
    #
    #     # 取得 feature
    #     with open(self.model2_train_feature_file_path) as f:
    #         reader = csv.reader(f)
    #         features = [row for row in reader]
    #
    #         """ release memory """
    #         del reader  # clean parameter
    #         gc.collect()  # 清除或釋放未引用的記憶體
    #
    #     # 取得 label
    #     with open(self.model2_train_label_file_path) as f:
    #         reader = csv.reader(f)
    #         labels = list(reader)
    #
    #         """ release memory """
    #         del reader  # clean parameter
    #         gc.collect()  # 清除或釋放未引用的記憶體
    #
    #     # CSV取得都是字串，要給numpy.array用就得改成int
    #     features = [[int(j) for j in i] for i in features]
    #     labels = [int(i[0]) for i in labels]
    #
    #     """ features & labels """
    #     train_x = numpy.array(features)
    #     train_y = numpy.array(labels)
    #
    #     train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    #
    #     train_dl = DataLoader(train_ds, shuffle=True, batch_size=self.p_batch_size, drop_last=True)
    #
    #     # TODO 要修改參數
    #     """ Model2 """
    #     if os.path.exists(self.model2_name):
    #         print('模組2已建立，匯入模型')
    #         # 從檔案中載入模型
    #         model2 = torch.load(self.model2_name)
    #         if os.path.exists(self.weights2_name):
    #             # 從檔案載入參數（不含模型）
    #             model2.load_state_dict(torch.load(self.weights2_name))
    #         model2.eval()
    #     else:
    #         print('無舊資料，建立新模組2')
    #         # 建構LSTM物件
    #         model2 = LSTMModel2(self.p_vocab_size, self.p_embedding_dim, self.p_hidden_dim, self.p_dropout)
    #         model2 = model2.to(self.device)
    #
    #     # 優化器
    #     optimizer = torch.optim.Adam(model2.parameters(), lr=self.p_learning_rate)
    #
    #     # 宣告損失函數
    #     criterion = nn.CrossEntropyLoss()
    #
    #     print(model2)
    #
    #     losses = []
    #
    #     for e in range(self.p_epochs):
    #
    #         h0, c0 = model2.init_hidden(self.p_batch_size, self.p_hidden_dim)
    #
    #         h0 = h0.to(self.device)
    #         c0 = c0.to(self.device)
    #
    #         for batch_idx, batch in enumerate(train_dl):
    #             input_ = batch[0].to(self.device)
    #             target_ = batch[1].to(self.device)
    #
    #             optimizer.zero_grad()  # 模型的參數梯度初始化為0
    #             with torch.set_grad_enabled(True):
    #                 out, hidden = model2(input_, (h0, c0))
    #                 loss = criterion(out, target_)
    #                 loss.backward()
    #                 optimizer.step()  # 更新所有參數
    #
    #         losses.append(loss.item())
    #         print(e + 1, loss.item())
    #
    #     """ save model & weight """
    #     torch.save(model2, self.model2_name)
    #     torch.save(model2.state_dict(), self.weights2_name)
    #
    #     """ check folder """
    #     if not os.path.exists(self.log_path):
    #         # 建立 LOG 主目錄
    #         os.makedirs(self.log_path)
    #     if not os.path.exists(self.back_up_path) and self.p_backup_flag:
    #         # 建立 back up 目錄
    #         os.makedirs(self.back_up_path)
    #
    #     """ 將PARAMETER寫進LOG PATH """
    #     parameter_file_name = self.log_path + 'parameter.json'
    #     parameter_json = {
    #         'model': 'model2',
    #         'remark': self.remark,
    #         'ticker': self.p_ticker,
    #         'start_date': self.p_start_date,
    #         'end_date': self.p_end_date,
    #         'hidden_dim': self.p_hidden_dim,
    #         'vocab_size': self.p_vocab_size,
    #         'batch_size': self.p_batch_size,
    #         'epochs': self.p_epochs,
    #         'learning_rate': self.p_learning_rate
    #     }
    #     with open(parameter_file_name, 'a') as p:
    #         json.dump(parameter_json, p)
    #
    #     # save history
    #     with open(self.log_name, 'a') as f:
    #         """ first row is title """
    #         log_title = ['index', 'loss']
    #         write = csv.writer(f)
    #         write.writerow(log_title)
    #
    #         """ history data """
    #         for i in range(len(losses)):
    #             row = [i, losses[i]]
    #             write = csv.writer(f)
    #             write.writerow(row)
    #
    #     # 通知
    #     remark_msg = "\n備註: " + self.remark
    #     param_msg = "\n訓練次數: {}".format(str(self.p_epochs))
    #     time_msg = self.function_calc_spend_time(start_time)
    #     line_notify_message('dissertation_04_Training_Model model2 step2 finished!' + remark_msg + param_msg + time_msg)

    def run_model2_step3(self):
        """
           預測並驗證 model2
        """
        pass

    def run_model2_step4(self):
        """
            使用非訓練時間的 dateset 做 predict
        """

        print('model2 step4 start')
        start_time = time.time()

        """ check model file """
        if not os.path.exists(self.model1_name):
            print('沒有 model1')
            quit()

        """'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
        # 從檔案中載入模型
        """ model1 """
        model1 = torch.load(self.model1_name)
        if os.path.exists(self.weights1_name):
            # 從檔案載入參數（不含模型）
            model1.load_state_dict(torch.load(self.weights1_name))
        model1.eval()
        """ model2 """
        model2 = torch.load(self.model2_name)
        if os.path.exists(self.weights2_name):
            # 從檔案載入參數（不含模型）
            model2.load_state_dict(torch.load(self.weights2_name))
        model2.eval()

        # 優化器
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.p_learning_rate)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=self.p_learning_rate)
        """'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""

        """ 讀取新聞 """
        test_news_key = {'news_id': '1986246'}  # TODO : TEST 新聞資料太少

        db_news_data = self.coll_analyze_news.find(test_news_key)

        news_data_list = []
        news_data_encoding_list = []
        for news_data in db_news_data:
            if news_data['ticker'] == self.p_ticker:
                news_data_list.append(news_data)

                """ get encoding data """
                news_encoding_key = {'sequence': news_data['sequence']}
                news_encoding_data = self.coll_analyze_news_encoding.find_one(news_encoding_key)
                if news_encoding_data:
                    news_data_encoding_list.append(news_encoding_data)

        train_list = []

        """ 讀取三句話 """
        for i in range(len(news_data_encoding_list) - 2):
            ''' create train dataset '''
            encoding_list = news_data_encoding_list[i]['news_encoding_list']
            encoding_list.extend(news_data_encoding_list[i + 1]['news_encoding_list'])
            encoding_list.extend(news_data_encoding_list[i + 2]['news_encoding_list'])

            """ padding """
            if len(encoding_list) > self.p_encoding_length:
                encoding_list = encoding_list[:self.p_encoding_length]
            else:
                encoding_list.extend(
                    list([0 for _ in range(self.p_encoding_length - len(encoding_list))])
                )

            train_list.append(encoding_list)

        ''' *********** predict test *********** '''

        # CSV取得都是字串，要給numpy.array用就得改成int
        features = [[int(j) for j in i] for i in train_list]
        labels = [int(0) for _ in range(len(train_list))]  # TODO

        """ features & labels """
        test_x = numpy.array(features)
        test_y = numpy.array(labels)

        test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        test_dl = DataLoader(test_ds, shuffle=True, batch_size=self.p_batch_size, drop_last=True)

        h0, c0 = model1.init_hidden(self.p_batch_size, self.p_hidden_dim)

        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        batch_acc = []
        for batch_idx, batch in enumerate(test_dl):
            input_ = batch[0].to(self.device)
            target_ = batch[1].to(self.device)

            optimizer1.zero_grad()
            with torch.set_grad_enabled(False):
                out, hidden = model1(input_, (h0, c0))
                _, preds = torch.max(out, 1)
                preds = preds.to("cpu").tolist()
                batch_acc.append(accuracy_score(preds, target_.tolist()))

        accuracy = sum(batch_acc) / len(batch_acc)

        print('accuracy:', accuracy)

        # 通知
        remark_msg = "\n備註: " + self.remark
        acc_msg = "\naccuracy: " + str(accuracy)
        time_msg = self.function_calc_spend_time(start_time)
        line_notify_message('dissertation_04_Training_Model model2 step4 finished!' + remark_msg + acc_msg + time_msg)

    @staticmethod
    def function_calc_spend_time(start_time):
        total_second = time.time() - start_time
        total_minute = total_second // 60
        show_hour = round(total_minute // 60)  # 時
        show_minute = round(total_minute % 60)  # 分
        show_second = round(total_second % 60)  # 秒
        time_msg = "\n花費時間: {} h {} m {} s".format(str(show_hour), str(show_minute), str(show_second))
        return time_msg

    @staticmethod
    def set_label_value(input_value):
        """
        設定 label 值
        """
        if input_value > 0:
            # 上取整函數
            return int(math.ceil(input_value))
        elif input_value < 0:
            # 下取整函數
            return int(math.floor(input_value))
        else:
            return 0


if __name__ == '__main__':
    execute = Entrance()
    execute.run()
