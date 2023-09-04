from Entrance import Entrance
from LineNotifyMessage import line_notify_message

p_loop_cnt = 3  # 重複執行次數

for training_time in range(p_loop_cnt):
    msg = "dissertation_04_Training_Model loop, 第{}次, 共{}次".format(str(training_time+1), str(p_loop_cnt))
    line_notify_message(msg)

    print('training time: {} start'.format(str(training_time + 1)))
    entrance = Entrance()
    entrance.run()
    print('training time: {} end'.format(str(training_time + 1)))
