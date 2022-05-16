# -*- coding: utf-8 -*-
import FileWrite
import config
import datetime
import os
if __name__ == "__main__":
    config = config.read_config()

    dataTime = datetime.datetime.strptime(config['startTime'], "%Y%m%d%H%M%S")
    path = os.path.join(config['TruthFileDir'], 'adtd'+ dataTime.strftime('_%Y_%m_%d') + '.txt')


    file_write = FileWrite.TxtTrueFiletoGrid(config)

    file_write.all_trueFile_txt_to_npy()

